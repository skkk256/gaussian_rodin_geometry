import argparse
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True

from pipeline import Job, Pipeline
from io_utils import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.job.read_text())
    parent = Path(request['parentWork'])
    config = json.loads((parent / 'config.json').read_text())
    assert sha256(Path(config['source'])) == config['sourceSha256'], 'Source Gaussian changed since the original run'
    root = args.job.parent
    work = root / 'work' / request['scene_id'] / request['object_id']
    work.mkdir(parents=True, exist_ok=True)
    reuse = ['canonical.json', 'gaussian.png']
    if request['mode'] == 'rodin':
        reuse += ['fal_refine.png', 'fal-request.json', 'gaussian_filter.ply', 'gaussian_filter.png', 'condition.npz']
    for name in reuse:
        assert (parent / name).is_file(), f'Missing reusable cache: {name}'
        shutil.copyfile(parent / name, work / name)
    job = Job(Path(config['source']), request['scene_id'], request['object_id'], request['seed'])
    pipeline = Pipeline(request['device'], Path(request['credentials']), bbox='bbox' in config)
    pipeline.run(job, root / 'result', root / 'work')


if __name__ == '__main__':
    main()
