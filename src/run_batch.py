import argparse
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True

from pipeline import PROJECT, Pipeline
from jobs import load_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Gaussian PLY manifest entries; result/<scene_id>/<object_id>/.')
    parser.add_argument('--manifest', type=Path, required=True, help='JSON list: scene_id, object_id, input, seed')
    parser.add_argument('--output', type=Path, default=PROJECT / 'result')
    parser.add_argument('--work-dir', type=Path, default=PROJECT / '.work')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--max-retry', type=int, default=1, help='Additional attempts below score 70; alternate Rodin-only and FAL+Rodin retries')
    args = parser.parse_args()
    jobs = load_jobs(args.manifest)
    pipeline = Pipeline(args.device, args.credentials, max_retry=args.max_retry)
    failures = []
    for job in jobs:
        try:
            pipeline.run(job, args.output, args.work_dir)
        except Exception:
            failures.append(f'{job.scene_id}/{job.object_id}')
            traceback.print_exc()
    assert not failures, f'Failed objects: {failures}; rerun the same command to resume saved requests'


if __name__ == '__main__':
    main()
