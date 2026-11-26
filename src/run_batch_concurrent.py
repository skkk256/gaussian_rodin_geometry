import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from pipeline import PROJECT, Pipeline
from concurrent_pipeline import ConcurrentBatch
from jobs import load_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description='Four-stage concurrent Gaussian → Rodin pipeline with persistent SAM3D.')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'result')
    parser.add_argument('--work-dir', type=Path, default=PROJECT / '.work')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--max-retry', type=int, default=1, help='Additional attempts below score 70; alternate Rodin-only and FAL+Rodin retries')
    parser.add_argument('--fal-workers', type=int, default=2)
    parser.add_argument('--rodin-workers', type=int, default=2)
    parser.add_argument('--max-inflight', type=int, default=8)
    args = parser.parse_args()
    jobs = load_jobs(args.manifest)
    pipeline = Pipeline(args.device, args.credentials, max_retry=args.max_retry)
    ConcurrentBatch(pipeline, args.fal_workers, args.rodin_workers, args.max_inflight).run(jobs, args.output, args.work_dir)


if __name__ == '__main__':
    main()
