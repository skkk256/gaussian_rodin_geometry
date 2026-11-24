import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True

from io_utils import OUTPUT_FILES, read_rgb, write_json

PROJECT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT / 'src/viewer'


class Results:
    def __init__(self, root: Path, work: Path, credentials: Path, device: str):
        self.root, self.work = root.resolve(), work.resolve()
        self.credentials, self.device = credentials.resolve(), device
        self.root.mkdir(parents=True, exist_ok=True)
        self.private = PROJECT / '.viewer' / hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
        self.private.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.catalog_lock = threading.Lock()
        self.index_path = self.private / 'catalog.json'
        self.index = json.loads(self.index_path.read_text()) if self.index_path.exists() else None
        self.preview_keys = dict(self.index['previews']) if self.index is not None else {}
        self.preview_locks = {}
        self.preview_slots = threading.Semaphore(4)
        self.previews = self.private / 'previews'
        self.previews.mkdir(exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}
        for path in self.private.glob('*/job.json'):
            job = json.loads(path.read_text())
            if job['status'] in ('queued', 'running'):
                job.update(status='interrupted', error='Viewer stopped; inspect the log before submitting a new run')
                write_json(path, job)
            self.jobs[job['id']] = job

    def selected(self, scene: str, obj: str, version: str) -> tuple[Path, Path]:
        assert all(v and v not in ('.', '..') and '/' not in v and '\\' not in v for v in (scene, obj))
        if version == 'original':
            result, work = self.root, self.work
        else:
            job = self.jobs[version]
            assert (job['scene_id'], job['object_id'], job['status']) == (scene, obj, 'completed')
            result, work = self.private / version / 'result', self.private / version / 'work'
        return result / scene / obj, work / scene / obj

    def index_object(self, folder: Path) -> tuple[dict, dict]:
        with os.scandir(folder) as entries:
            files = {entry.name: entry for entry in entries}
        assert all(name in files for name in OUTPUT_FILES), f'Incomplete result: {folder}'
        previews = {}
        for name in OUTPUT_FILES:
            if name.endswith('.png'):
                info = files[name].stat()
                source = folder / name
                previews[str(source)] = hashlib.sha256(f'{source}:{info.st_mtime_ns}:{info.st_size}:768:aspect:85'.encode()).hexdigest()
        scene, obj = folder.parent.name, folder.name
        revision = hashlib.sha256(''.join(previews.values()).encode()).hexdigest()[:16]
        return dict(scene_id=scene, object_id=obj, previewRevision=revision,
                    hasWork=(self.work / scene / obj / 'config.json').is_file()), previews

    def catalog(self, refresh: bool = False) -> dict:
        with self.catalog_lock:
            if refresh or self.index is None:
                folders = sorted(path.parent for path in self.root.glob('*/*/transform.json'))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    entries = list(pool.map(self.index_object, folders))
                self.index = dict(objects=[item for item, _ in entries],
                                  previews={source: key for _, previews in entries for source, key in previews.items()})
                self.preview_keys = dict(self.index['previews'])
                write_json(self.index_path, self.index)
            indexed = self.index['objects']
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        credentials = self.credentials.is_file()
        objects = []
        for item in indexed:
            scene, obj = item['scene_id'], item['object_id']
            versions = [dict(id='original', label='首次结果')]
            versions.extend(dict(id=j['id'], label='重跑 ' + j['id']) for j in jobs
                            if (j['scene_id'], j['object_id'], j['status']) == (scene, obj, 'completed'))
            objects.append(dict(scene_id=scene, object_id=obj, versions=versions,
                                previewRevision=item['previewRevision'], canRetry=item['hasWork'] and credentials))
        return dict(objects=objects, jobs=jobs, token=self.token)

    def preview(self, source: Path) -> Path:
        import cv2

        with self.lock:
            lock = self.preview_locks.setdefault(str(source), threading.Lock())
        with lock:
            if str(source) not in self.preview_keys:
                info = source.stat()
                self.preview_keys[str(source)] = hashlib.sha256(f'{source}:{info.st_mtime_ns}:{info.st_size}:768:aspect:85'.encode()).hexdigest()
            target = self.previews / (self.preview_keys[str(source)] + '.webp')
            if target.exists():
                return target
            with self.preview_slots:
                rgb = read_rgb(source)
                rgb = cv2.resize(rgb, (768, round(768 * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.webp', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_WEBP_QUALITY, 85])
                assert ok
                temporary = target.with_name('.' + target.name + '.' + secrets.token_hex(6))
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(target)
        return target

    def submit(self, request: dict) -> dict:
        scene, obj, version, mode = (request[k] for k in ('scene_id', 'object_id', 'version', 'mode'))
        assert mode in ('rodin', 'full')
        result, work = self.selected(scene, obj, version)
        assert all((result / name).is_file() for name in OUTPUT_FILES), 'Result is not complete'
        assert (work / 'config.json').is_file(), 'Original work directory is required for retries'
        assert self.credentials.is_file(), 'Provide --credentials to enable retries'
        with self.lock:
            assert not any(j['status'] in ('queued', 'running') for j in self.jobs.values()), 'A retry is already queued or running'
            identifier = time.strftime('%Y%m%d-%H%M%S') + '-' + secrets.token_hex(3)
            job = dict(id=identifier, scene_id=scene, object_id=obj, mode=mode,
                       seed=secrets.randbelow(65536), status='queued', createdAt=time.time(),
                       parentWork=str(work), credentials=str(self.credentials), device=self.device)
            folder = self.private / identifier
            folder.mkdir()
            (folder / 'run.log').touch()
            write_json(folder / 'job.json', job)
            self.jobs[identifier] = job
            self.pool.submit(self.run, identifier)
            return dict(job)

    def run(self, identifier: str) -> None:
        folder = self.private / identifier
        with self.lock:
            self.jobs[identifier]['status'] = 'running'
            write_json(folder / 'job.json', self.jobs[identifier])
        try:
            with (folder / 'run.log').open('ab') as log:
                process = subprocess.run([sys.executable, '-B', '-u', str(PROJECT / 'src/viewer_worker.py'),
                                          '--job', str(folder / 'job.json')], stdout=log, stderr=subprocess.STDOUT)
            assert process.returncode == 0, f'Worker exited with {process.returncode}; open the run log'
        except Exception as error:
            status, message = 'failed', str(error)
        else:
            status, message = 'completed', ''
        with self.lock:
            self.jobs[identifier].update(status=status, error=message)
            write_json(folder / 'job.json', self.jobs[identifier])


class Handler(SimpleHTTPRequestHandler):
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map,
                          **{'.log': 'text/plain; charset=utf-8', '.webp': 'image/webp'})

    def __init__(self, *args, results: Results, **kwargs):
        self.results = results
        self.cache_control = 'no-store'
        super().__init__(*args, directory=str(ASSETS), **kwargs)

    def json(self, value: dict, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        route = unquote(urlsplit(self.path).path)
        parts = route.strip('/').split('/')
        try:
            if route == '/api/objects':
                self.json(self.results.catalog(refresh=urlsplit(self.path).query == 'refresh=1'))
            elif len(parts) == 5 and parts[0] == 'preview':
                _, version, scene, obj, name = parts
                assert name in ('gaussian.webp', 'fal_refine.webp', 'gaussian_filter.webp', 'rodin.webp')
                folder, _ = self.results.selected(scene, obj, version)
                self.file(self.results.preview(folder / Path(name).with_suffix('.png')))
            elif len(parts) == 5 and parts[0] == 'result':
                _, version, scene, obj, name = parts
                assert name in OUTPUT_FILES
                folder, _ = self.results.selected(scene, obj, version)
                self.file(folder / name)
            elif len(parts) == 5 and parts[0:2] == ['api', 'frame']:
                _, _, version, scene, obj = parts
                folder, _ = self.results.selected(scene, obj, version)
                from gaussian import compute_semantic_obb, read_ply
                xyz = read_ply(folder / 'gaussian_filter.ply')['xyz']
                transform = json.loads((folder / 'transform.json').read_text())
                if len(xyz) == 0:
                    assert transform['imageOnly']
                    obb = transform['referenceObb']
                else:
                    obb = compute_semantic_obb(xyz)
                self.json(dict(obb=obb, canonicalToWorld=transform['rodinToWorld']['matrix'], gaussianCount=len(xyz)))
            elif len(parts) == 4 and parts[0:2] == ['api', 'jobs'] and parts[3] == 'log':
                assert parts[2] in self.results.jobs
                folder = self.results.private / parts[2]
                self.file(folder / 'run.log')
            elif route in ('/', '/index.html', '/gallery.js', '/style.css', '/compare.html', '/compare.js'):
                self.file(ASSETS / ('index.html' if route == '/' else parts[0]))
            elif parts[0] == 'vendor':
                path = (PROJECT / 'third_party/viewer' / Path(*parts[1:])).resolve()
                assert path.is_relative_to(PROJECT / 'third_party/viewer')
                self.file(path)
            else:
                self.send_error(404)
        except (AssertionError, KeyError, FileNotFoundError) as error:
            self.json(dict(error=str(error)), 404)

    def file(self, path: Path) -> None:
        info = path.stat()
        if path.suffix in ('.png', '.webp'):
            self.cache_control = 'private, max-age=3600'
            etag = f'"{info.st_mtime_ns:x}-{info.st_size:x}"'
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.send_header('ETag', etag)
                self.end_headers()
                return
        self.send_response(200)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(info.st_size))
        if path.suffix in ('.png', '.webp'):
            self.send_header('ETag', etag)
        self.end_headers()
        with path.open('rb') as stream:
            self.copyfile(stream, self.wfile)

    def do_POST(self) -> None:
        if self.path != '/api/retry':
            self.send_error(404)
            return
        try:
            assert self.headers.get('X-Retry-Token') == self.results.token, 'Reload the page before retrying'
            assert urlsplit(self.headers.get('Origin', '')).netloc == self.headers['Host'], 'Invalid request origin'
            length = int(self.headers.get('Content-Length', '0'))
            assert 0 < length <= 4096
            self.json(self.results.submit(json.loads(self.rfile.read(length))), 202)
        except (AssertionError, KeyError, ValueError) as error:
            self.json(dict(error=str(error)), 400)

    def end_headers(self) -> None:
        self.send_header('Cache-Control', self.cache_control)
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description='Browse result/<scene_id>/<object_id> and rerun selected objects.')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=PROJECT / '.work')
    parser.add_argument('--credentials', type=Path, default=PROJECT / '.env.example')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=7888)
    args = parser.parse_args()
    results = Results(args.results, args.work_dir, args.credentials, args.device)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, results=results))
    print(f'Results: {results.root}\nViewer: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
