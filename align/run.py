"""Run alignment only; generation and texturing have already finished."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'

import argparse
import fcntl
import gc
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from align import PROJECT
from align.matching import DinoMatcher, MATCHING_VERSION
from align.pipeline import align_case, input_hashes_for
from align.publish import publish
from conditions import PointSampler
from io_utils import sha256, write_json
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description='Saved inverse -> filtered-Gaussian FPS bbox -> eight-view DINOv3/SIFT; no generation API calls.')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--result', type=Path, help='One object result directory containing rodin_textured.glb, transform.json and gaussian_filter.ply')
    inputs.add_argument('--manifest', type=Path, help='Generation manifest; uses scene_id/object_id to locate results and work directories')
    parser.add_argument('--results-root', type=Path, default=PROJECT/'result', help='Batch generation result root')
    parser.add_argument('--work-root', type=Path, default=PROJECT/'.work', help='Generation work root')
    parser.add_argument('--work-dir', type=Path, help='Single object work directory; default: work-root/<scene>/<object>')
    parser.add_argument('--output', type=Path, default=PROJECT/'align/output')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--weights', type=Path, default=PROJECT/'ckpt/dinov3')
    parser.add_argument('--texture-slot', choices=('emissive', 'base-color'), default='emissive', help='Explicit glTF texture slot used for matching')
    parser.add_argument('--per-view-framing', action='store_true', help='Fit camera distance independently for each view, useful for long thin objects')
    parser.add_argument('--resume', action='store_true', help='Reuse only completed outputs with identical inputs, code, weights and settings')
    args = parser.parse_args()
    device = torch.device(args.device)
    assert device.type == 'cuda' and device.index is not None, 'Use an indexed CUDA device, e.g. cuda:1'
    assert device.index < torch.cuda.device_count()
    if args.result is not None:
        result = args.result.resolve(strict=True)
        key = result.parts[-2:]
        work = args.work_dir.resolve(strict=True) if args.work_dir is not None else args.work_root.joinpath(*key).resolve(strict=True)
        jobs = [(result, work)]
    else:
        assert args.work_dir is None, '--work-dir is only for --result'
        entries = json.loads(args.manifest.read_text())
        assert isinstance(entries, list) and entries
        keys = [(row['scene_id'], row['object_id']) for row in entries]
        assert len(set(keys)) == len(keys), 'Duplicate scene_id/object_id'
        assert all(part and part not in ('.', '..') and '/' not in part and '\\' not in part for key in keys for part in key)
        jobs = [(args.results_root.joinpath(*key).resolve(), args.work_root.joinpath(*key).resolve()) for key in keys]
    output = args.output.resolve()
    assert not any(output == result or output.is_relative_to(result) or result.is_relative_to(output)
                   or output == work or output.is_relative_to(work) or work.is_relative_to(output) for result, work in jobs), 'Alignment output must be separate from generation result/work'
    output.mkdir(parents=True, exist_ok=True)
    code_paths = sorted(Path(__file__).parent.glob('*.py'))+[PROJECT/'src/conditions.py', PROJECT/'src/gaussian.py']
    signature = dict(code={path.name: sha256(path) for path in code_paths},
                     weights=str(args.weights.resolve(strict=True)),
                     weightFiles={path.name: [path.stat().st_size, path.stat().st_mtime_ns] for path in args.weights.iterdir() if path.is_file()})
    settings = dict(perViewFraming=args.per_view_framing, textureSlot=args.texture_slot)
    matcher, sampler = None, PointSampler(str(device))
    failures = []
    for result, work in jobs:
        case = '/'.join(result.parts[-2:])
        folder = output.joinpath(*result.parts[-2:])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                hashes = input_hashes_for(result, work)
                if args.resume and (folder/'result.json').is_file():
                    report = json.loads((folder/'result.json').read_text())
                    assert report['inputHashes'] == hashes and report['settings'] == settings and report['runtime'] == signature and report['matchingVersion'] == MATCHING_VERSION, 'Cached inputs/code/settings changed; rerun without --resume'
                    assert report['outputHashes'] == {name: sha256(folder/name) for name in report['outputHashes']}, 'Cached alignment output changed; rerun without --resume'
                    print(case+': already completed', flush=True)
                else:
                    (folder/'result.json').unlink(missing_ok=True)
                    if matcher is None:
                        matcher = DinoMatcher(args.weights, device)
                    report = align_case(result, work, folder, matcher, sampler, device,
                                        per_view=args.per_view_framing, texture_slot=args.texture_slot)
                    report['runtime'] = signature
                    assets = {'original-gaussian.ply': Path(json.loads((work/'config.json').read_text())['source']),
                              'filtered-gaussian.ply': result/'gaussian_filter.ply', 'model.glb': result/'rodin_textured.glb'}
                    for name, target in assets.items():
                        link = folder/name
                        assert not link.exists() or link.is_symlink(), link
                        link.unlink(missing_ok=True)
                        link.symlink_to(target.resolve(strict=True))
                    report['outputHashes'] = {path.name: sha256(path) for path in sorted(folder.iterdir()) if path.is_file() and path.name not in ('run.lock', 'result.json', 'error.json')}
                    write_json(folder/'result.json', report)
                (folder/'error.json').unlink(missing_ok=True)
            except Exception as error:
                failures.append(case)
                (folder/'result.json').unlink(missing_ok=True)
                write_json(folder/'error.json', dict(case=case, status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc()))
                traceback.print_exc()
            publish(output)
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    print(f'Alignment viewer: {output}/index.html', flush=True)
    assert not failures, f'Alignment failed for {failures}; tracebacks are saved in error.json'


if __name__ == '__main__':
    main()
