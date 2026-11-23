import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from pipeline import Job, PROJECT, Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description='Gaussian PLY -> Fal four views -> filtered FPS -> Rodin + complete inverse transform.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--object-id', required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'result')
    parser.add_argument('--work-dir', type=Path, default=PROJECT / '.work')
    parser.add_argument('--device', default='cuda:1', help='Indexed CUDA device used by SAM3D, rendering and FPS')
    parser.add_argument('--credentials', type=Path, help='Private dotenv containing FAL_API_KEY and RODIN_API_KEY; otherwise use environment variables')
    parser.add_argument('--seed', type=int, default=22101)
    parser.add_argument('--bbox', action='store_true', help='After inverse transform, fit uniform scale and translation to the filtered-Gaussian FPS bbox; save the combined transform')
    args = parser.parse_args()
    job = Job(args.input.resolve(strict=True), args.scene_id, args.object_id, args.seed)
    job.validate()
    Pipeline(args.device, args.credentials, bbox=args.bbox).run(job, args.output, args.work_dir)


if __name__ == '__main__':
    main()
