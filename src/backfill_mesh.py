import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True

from pipeline import PROJECT
from io_utils import OUTPUT_FILES, join_views, sha256, write_json, write_rgb
from mesh_render import MeshRenderer
from timing import Timing
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Add aligned Rodin six-view images to completed results; no generation API calls.')
    parser.add_argument('--results', type=Path, default=PROJECT / 'result')
    parser.add_argument('--work-dir', type=Path, default=PROJECT / '.work')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    renderer = MeshRenderer(args.device)
    count = 0
    for output in sorted(args.results.resolve().glob('*/*')):
        if not output.is_dir() or (output / 'rodin.png').exists():
            continue
        work = args.work_dir.resolve() / output.parent.name / output.name
        with (work / 'run.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'{output.parent.name}/{output.name}: running; left unchanged', flush=True)
                continue
            state = json.loads((work / 'state.json').read_text())
            assert state['status'] == 'completed', output
            assert set(path.name for path in output.iterdir()) == set(OUTPUT_FILES) - {'rodin.png'}
            assert state['outputSha256'] == {name: sha256(output / name) for name in state['outputSha256']}
            timing = Timing(work / 'timing.json', time.perf_counter(), 0.)
            timing.attempt['kind'] = 'mesh_backfill'
            with timing.stage('mesh_render'):
                frame = json.loads((work / 'canonical.json').read_text())
                transform = json.loads((output / 'transform.json').read_text())
                matrix = np.asarray(transform['rodinToWorld']['matrix']).reshape(4, 4, order='F')
                write_rgb(work / 'rodin.png', join_views(renderer.render(output / 'rodin.glb', matrix, frame)))
                shutil.copyfile(work / 'rodin.png', output / 'rodin.png')
            timing.finish('completed')
            write_json(output / 'timing.json', timing.data)
            state['outputSha256'] = {name: sha256(output / name) for name in OUTPUT_FILES}
            write_json(work / 'state.json', state)
            count += 1
            print(f'{output.parent.name}/{output.name}: rodin.png added in {timing.attempt["totalSeconds"]:.2f}s', flush=True)
    print(f'Added mesh previews to {count} completed objects.', flush=True)


if __name__ == '__main__':
    main()
