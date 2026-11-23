import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True

from io_utils import OUTPUT_FILES, write_json

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

    def catalog(self) -> dict:
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        objects = []
        for transform in sorted(self.root.glob('*/*/transform.json')):
            folder = transform.parent
            assert all((folder / name).is_file() for name in OUTPUT_FILES), f'Incomplete result: {folder}'
            scene, obj = folder.parent.name, folder.name
            versions = [dict(id='original', label='首次结果')]
            versions.extend(dict(id=j['id'], label='重跑 ' + j['id']) for j in jobs
                            if (j['scene_id'], j['object_id'], j['status']) == (scene, obj, 'completed'))
            objects.append(dict(scene_id=scene, object_id=obj, versions=versions,
                                canRetry=(self.work / scene / obj / 'config.json').is_file() and self.credentials.is_file()))
        return dict(objects=objects, jobs=jobs, token=self.token)

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
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map, **{'.log': 'text/plain; charset=utf-8'})

    def __init__(self, *args, results: Results, **kwargs):
        self.results = results
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
                self.json(self.results.catalog())
            elif len(parts) == 5 and parts[0] == 'result':
                _, version, scene, obj, name = parts
                assert name in OUTPUT_FILES
                folder, _ = self.results.selected(scene, obj, version)
                self.file(folder / name)
            elif len(parts) == 5 and parts[0:2] == ['api', 'frame']:
                _, _, version, scene, obj = parts
                folder, _ = self.results.selected(scene, obj, version)
                from gaussian import compute_semantic_obb, read_ply
                obb = compute_semantic_obb(read_ply(folder / 'gaussian_filter.ply')['xyz'])
                transform = json.loads((folder / 'transform.json').read_text())
                self.json(dict(obb=obb, canonicalToWorld=transform['rodinToWorld']['matrix']))
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
        assert path.is_file(), path.name
        self.send_response(200)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(path.stat().st_size))
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
        self.send_header('Cache-Control', 'no-store')
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
