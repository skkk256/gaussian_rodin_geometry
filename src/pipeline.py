from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'third_party'))
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ['DINOV2_CHECKPOINT'] = str(PROJECT / 'ckpt/dinov2_vitl14_reg4_pretrain.pth')

from dataclasses import dataclass
import fcntl
import json
import shutil
import tempfile
import time
from typing import TextIO

import numpy as np
from dotenv import load_dotenv
import torch

from api import FAL_MODEL, FAL_SETTINGS, NANO_MODEL, NANO_SETTINGS, RODIN_SETTINGS, GenerationAPI
from bbox import register_bbox
from conditions import PointSampler, filter_foreground, normalize_points
from foreground import ForegroundSegmenter
from gaussian import compute_semantic_obb, covariance_prefilter, read_ply, write_ply
from io_utils import OUTPUT_FILES, join_views, read_rgb, sha256, split_views, validate_glb, write_json, write_rgb
from orientation import Orientation
from render import FOUR_ANGLES, GaussianRenderer
from timing import Timing
from mesh_render import MeshRenderer
from resources import Resource

MIN_FOREGROUND_RETAINED_FRACTION = .30


@dataclass(frozen=True)
class Job:
    source: Path
    scene_id: str
    object_id: str
    seed: int

    def validate(self) -> None:
        assert self.source.is_file() and self.source.suffix.lower() == '.ply', self.source
        assert all(value and value not in ('.', '..') and '/' not in value and chr(92) not in value
                   for value in (self.scene_id, self.object_id)), 'scene_id/object_id must be single directory names'
        assert 0 <= self.seed < 65536


@dataclass
class Task:
    job: Job
    output: Path
    work: Path
    timing: Timing
    lock: TextIO
    queued_at: float
    prepared: tuple[dict, dict] | None = None


class Pipeline:
    def __init__(self, device: str, credentials: Path | None, bbox: bool = False):
        started = time.perf_counter()
        self.bbox = bbox
        if credentials is not None:
            assert credentials.is_file(), credentials
            load_dotenv(credentials, override=True)
        self.api = GenerationAPI()
        self.renderer = GaussianRenderer(device)
        self.mesh_renderer = MeshRenderer(device)
        self.sampler = PointSampler(device)
        self.orientation = None
        self.segmenter = None
        self.gpu = Resource('gpu')
        self.segmentation = Resource('segmentation')
        self.prompt = (PROJECT / 'src/fal_prompt.txt').read_text()
        torch.set_num_threads(2)
        assert self.renderer.device.index < torch.cuda.device_count()
        assert all((PROJECT / 'ckpt' / name).is_file() for name in (
            'sam3d/pipeline_ssonly.yaml', 'sam3d/ss_generator.yaml', 'sam3d/ss_generator.ckpt',
            'sam3d/ss_decoder.yaml', 'sam3d/ss_decoder.ckpt', 'dinov2_vitl14_reg4_pretrain.pth',
            'u2net.onnx', 'isnet-general-use.onnx'))
        self.initialization_seconds = time.perf_counter() - started

    def begin(self, job: Job, output_root: Path, work_root: Path, *, flows: bool = False) -> Task | None:
        started = time.perf_counter()
        job.validate()
        output = output_root.resolve() / job.scene_id / job.object_id
        work = work_root.resolve() / job.scene_id / job.object_id
        assert not output.is_relative_to(work_root.resolve()) and not work.is_relative_to(output_root.resolve()), 'Keep work and output directories separate'
        work.mkdir(parents=True, exist_ok=True)
        lock = (work / 'run.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            config = dict(version=1, source=str(job.source.resolve()), sourceSha256=sha256(job.source),
                          sceneId=job.scene_id, objectId=job.object_id, seed=job.seed,
                          fal=dict(model=FAL_MODEL, settings=FAL_SETTINGS, prompt=self.prompt,
                                   retry=dict(model=NANO_MODEL, settings=dict(NANO_SETTINGS, seed=job.seed),
                                              minimumRetainedFraction=MIN_FOREGROUND_RETAINED_FRACTION)),
                          rodin=RODIN_SETTINGS)
            if self.bbox:
                config['bbox'] = True
            config_path = work / 'config.json'
            if config_path.exists():
                assert json.loads(config_path.read_text()) == config, 'Inputs/settings changed; choose a new work directory'
            else:
                assert not output.exists(), 'Existing output needs its original work directory to resume'
                write_json(config_path, config)
            state_path = work / 'state.json'
            if state_path.exists():
                previous = json.loads(state_path.read_text())
                if previous['status'] == 'completed':
                    assert set(path.name for path in output.iterdir()) == set(OUTPUT_FILES)
                    assert previous['outputSha256'] == {name: sha256(output / name) for name in OUTPUT_FILES}
                    print(f'{job.scene_id}/{job.object_id}: already completed', flush=True)
                    lock.close()
                    return None
            timing = Timing(work / 'timing.json', started, self.initialization_seconds, flows=flows)
            self.initialization_seconds = 0.
            return Task(job, output, work, timing, lock, started)
        except BaseException:
            lock.close()
            raise

    def run(self, job: Job, output_root: Path, work_root: Path) -> Path:
        task = self.begin(job, output_root, work_root)
        if task is None:
            return output_root.resolve() / job.scene_id / job.object_id
        try:
            self.prepare(task)
            self.refine(task, self.api)
            self.generate(task, self.api)
            self.postprocess(task)
            self.complete(task)
        except BaseException:
            self.fail(task)
            raise
        finally:
            task.lock.close()
        return task.output

    def phase(self, task: Task, message: str) -> None:
        print(f'{task.job.scene_id}/{task.job.object_id}: {message}', flush=True)
        write_json(task.work / 'state.json', dict(status='running', stage=message))

    def prepare(self, task: Task) -> None:
        work, timing = task.work, task.timing
        with timing.stage('read_and_prefilter'):
            data, _ = covariance_prefilter(read_ply(task.job.source))
        self.phase(task, 'SAM3D five-view orientation and Gaussian four-view render')
        frame_path = work / 'canonical.json'
        gaussian_image = work / 'gaussian.png'
        with self.gpu.use(timing):
            with timing.stage('sam3d_orientation', cached=frame_path.exists()):
                if not frame_path.exists():
                    if self.orientation is None:
                        self.orientation = Orientation(self.renderer, PROJECT / 'ckpt')
                    write_json(frame_path, self.orientation.prepare(data))
                frame = json.loads(frame_path.read_text())
            with timing.stage('gaussian_render', cached=gaussian_image.exists()):
                if not gaussian_image.exists():
                    write_rgb(gaussian_image, join_views(self.renderer.render(data, frame, FOUR_ANGLES, 1024)))
        task.prepared = data, frame

    def refine(self, task: Task, api: GenerationAPI) -> None:
        work, timing = task.work, task.timing
        assert task.prepared is not None
        data, frame = task.prepared
        filtered_path = work / 'gaussian_filter.ply'
        if filtered_path.exists():
            with timing.stage('fal_refine', cached=True):
                state = json.loads((work / 'fal-request.json').read_text())
                assert state['status'] == 'completed' and state['outputSha256'] == sha256(work / 'fal_refine.png')
            with timing.stage('foreground_filter', cached=True):
                assert len(read_ply(filtered_path)['xyz']) > 0
        else:
            selection = dict(beforeCount=len(data['xyz']), minimumRetainedFraction=MIN_FOREGROUND_RETAINED_FRACTION,
                             selectedModel=None, attempts=[])
            for model, directory, seed in ((FAL_MODEL, 'fal_gpt', None), (NANO_MODEL, 'fal_nano', task.job.seed)):
                attempt = work / directory
                attempt.mkdir(exist_ok=True)
                self.phase(task, f'{model}: four-view refinement')
                with timing.stage('fal_refine' if seed is None else 'fal_nano_retry', cached=(attempt / 'fal_refine.png').exists()):
                    fal_image = api.refine(work / 'gaussian.png', self.prompt, attempt, nano_seed=seed)
                    views = split_views(read_rgb(fal_image))
                with self.segmentation.use(timing):
                    with timing.stage('foreground_filter' if seed is None else 'foreground_filter_nano'):
                        self.phase(task, 'U2Net + ISNet foreground masks and single-view background veto')
                        if self.segmenter is None:
                            self.segmenter = ForegroundSegmenter(PROJECT / 'ckpt/u2net.onnx', PROJECT / 'ckpt/isnet-general-use.onnx')
                        alpha, _, _ = self.segmenter.multiview(views)
                        filtered = filter_foreground(data, frame, alpha)
                retained = len(filtered['xyz']) / len(data['xyz'])
                accepted = retained >= MIN_FOREGROUND_RETAINED_FRACTION
                state = json.loads((attempt / 'fal-request.json').read_text())
                selection['attempts'].append(dict(model=model, requestId=state['job']['request_id'],
                                                  afterCount=len(filtered['xyz']), retainedFraction=retained,
                                                  status='selected' if accepted else 'rejected'))
                if accepted:
                    selection['selectedModel'] = model
                write_json(work / 'foreground_filter.json', selection)
                self.phase(task, f'{model}: retained {len(filtered["xyz"])}/{len(data["xyz"])} Gaussians '
                                f'({retained:.2%}); {"accepted" if accepted else "below 30% retention"}')
                if not accepted:
                    if seed is None:
                        self.phase(task, 'More than 70% removed; retrying once with Nano Banana 2.1 using the original input')
                    continue
                shutil.copyfile(fal_image, work / 'fal_refine.png')
                shutil.copyfile(attempt / 'fal-request.json', work / 'fal-request.json')
                temporary = work / '.gaussian_filter.ply'
                write_ply(task.job.source, filtered['records'], temporary)
                temporary.replace(filtered_path)
                break
            assert filtered_path.exists(), 'GPT and Nano Banana both retained fewer than 30% of Gaussians; inspect foreground_filter.json'
        canonical_to_world = np.asarray(frame['canonicalToWorld']).reshape(4, 4, order='F')
        filtered_image = work / 'gaussian_filter.png'
        condition_path = work / 'condition.npz'
        with self.gpu.use(timing):
            with timing.stage('filtered_gaussian_render', cached=filtered_image.exists()):
                if not filtered_image.exists():
                    self.phase(task, 'Filtered Gaussian four-view render with the original cameras')
                    write_rgb(filtered_image, join_views(self.renderer.render(read_ply(filtered_path), frame, FOUR_ANGLES, 1024)))
            with timing.stage('sampling_fps_normalization', cached=condition_path.exists()):
                if not condition_path.exists():
                    self.phase(task, 'Gaussian volume sampling, FPS 4096, independent normalization to [-0.98, 0.98]')
                    canonical_points = self.sampler.sample(read_ply(filtered_path), canonical_to_world)
                    condition, normalization = normalize_points(canonical_points)
                    temporary = work / '.condition.npz'
                    np.savez(temporary, canonical=canonical_points, points=condition, normalization=normalization)
                    temporary.replace(condition_path)
        task.prepared = None

    def generate(self, task: Task, api: GenerationAPI) -> None:
        work, timing = task.work, task.timing
        self.phase(task, 'Rodin Gen-2.5-Medium / medium / point-cloud weight 0.3')
        with timing.stage('rodin_generate', cached=(work / 'rodin.glb').exists()):
            with np.load(work / 'condition.npz') as saved:
                condition = saved['points']
            views = split_views(read_rgb(work / 'fal_refine.png'))
            raw_mesh = api.rodin(views, condition, task.job.seed, work)
            validate_glb(raw_mesh)

    def postprocess(self, task: Task) -> None:
        work, timing = task.work, task.timing
        with timing.stage('inverse_transform'):
            frame = json.loads((work / 'canonical.json').read_text())
            canonical_to_world = np.asarray(frame['canonicalToWorld']).reshape(4, 4, order='F')
            with np.load(work / 'condition.npz') as saved:
                canonical_points, condition, normalization = saved['canonical'], saved['points'], saved['normalization']
            matrix = canonical_to_world @ np.linalg.inv(normalization)
            restored = condition @ matrix[:3, :3].T + matrix[:3, 3]
            world = canonical_points @ canonical_to_world[:3, :3].T + canonical_to_world[:3, 3]
            assert np.allclose(restored, world, atol=1e-6, rtol=1e-6), 'Condition inverse does not restore the sampled world points'
        with timing.stage('bbox_registration', skipped=not self.bbox):
            if self.bbox:
                self.phase(task, 'Fixed-rotation bbox registration to unnormalized filtered-Gaussian FPS')
                obb = compute_semantic_obb(read_ply(work / 'gaussian_filter.ply')['xyz'])
                matrix = register_bbox(work / 'rodin.glb', world, matrix, obb)
        with timing.stage('write_transform'):
            write_json(work / 'transform.json', dict(matrixOrder='column-major',
                       rodinToWorld=dict(matrix=matrix.flatten(order='F').tolist())))
        image = work / 'rodin.png'
        with self.gpu.use(timing):
            with timing.stage('mesh_render', cached=image.exists()):
                if not image.exists():
                    self.phase(task, 'Aligned Rodin mesh four-view render')
                    write_rgb(image, join_views(self.mesh_renderer.render(work / 'rodin.glb', matrix, frame)))
        self.phase(task, 'Export eight deliverables')
        with timing.stage('export'):
            task.output.parent.mkdir(parents=True, exist_ok=True)
            if task.output.exists():
                assert set(path.name for path in task.output.iterdir()) == set(OUTPUT_FILES)
                assert all(sha256(task.output / name) == sha256(work / name) for name in OUTPUT_FILES if name != 'timing.json'), 'Existing output differs from this task'
            else:
                with tempfile.TemporaryDirectory(prefix='.' + task.output.name + '-', dir=task.output.parent) as staging:
                    for name in OUTPUT_FILES:
                        shutil.copyfile(work / name, Path(staging) / name)
                    Path(staging).rename(task.output)

    def complete(self, task: Task) -> None:
        task.timing.finish('completed')
        write_json(task.output / 'timing.json', task.timing.data)
        write_json(task.work / 'state.json', dict(status='completed', seconds=task.timing.attempt['totalSeconds'],
                   outputSha256={name: sha256(task.output / name) for name in OUTPUT_FILES}))
        print(f'{task.job.scene_id}/{task.job.object_id}: completed in {task.timing.attempt["totalSeconds"]:.1f}s -> {task.output}', flush=True)

    def fail(self, task: Task) -> None:
        task.timing.finish('failed')
        write_json(task.work / 'state.json', dict(status='failed', stage=task.timing.attempt['stages'][-1]['name']))
