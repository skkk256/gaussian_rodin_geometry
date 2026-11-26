from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / 'third_party'))
sys.path.insert(0, str(PROJECT))
os.environ['NUMPY_MADVISE_HUGEPAGE'] = '0'
os.environ['LIDRA_SKIP_INIT'] = 'true'
os.environ['DINOV2_CHECKPOINT'] = str(PROJECT / 'ckpt/dinov2_vitl14_reg4_pretrain.pth')

from dataclasses import dataclass, field
import fcntl
import json
import shutil
import tempfile
import time
from typing import TextIO

import numpy as np
from dotenv import load_dotenv
import torch

from api import FAL_MODEL, FAL_SETTINGS, NANO_MODEL, NANO_SETTINGS, RODIN_SETTINGS, RODIN_VIEW_INDICES, GenerationAPI
from align.generation import ALIGNMENT_SETTINGS, MINIMUM_SCORE, align_generation
from align.matching import DinoMatcher
from alignment_worker import register_generation
from conditions import PointSampler, filter_foreground, normalize_points
from foreground import ForegroundSegmenter
from gaussian import covariance_prefilter, read_ply, write_ply
from io_utils import OUTPUT_FILES, join_views, read_rgb, sha256, split_views, validate_glb, write_json, write_rgb
from orientation import Orientation
from render import GaussianRenderer
from views import SIX_ANGLES, VIEW_SIZE
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
    image_only: bool = False
    attempt_index: int = 0
    candidates: list[dict] = field(default_factory=list)
    retry_phase: int | None = None

    @property
    def reference_work(self) -> Path:
        return self.work / 'refinements' / f'{self.attempt_index // 2:03d}'

    @property
    def generation_work(self) -> Path:
        return self.work / 'attempts' / f'{self.attempt_index:03d}'


class Pipeline:
    def __init__(self, device: str, credentials: Path | None, max_retry: int = 1):
        started = time.perf_counter()
        assert max_retry >= 0
        self.max_retry = max_retry
        if credentials is not None:
            assert credentials.is_file(), credentials
            load_dotenv(credentials, override=True)
        self.api = GenerationAPI()
        self.renderer = GaussianRenderer(device)
        self.mesh_renderer = MeshRenderer(device)
        self.sampler = PointSampler(device)
        self.orientation = None
        self.segmenter = None
        self.matcher = None
        self.gpu = Resource('gpu')
        self.segmentation = Resource('segmentation')
        self.prompt = (PROJECT / 'src/fal_prompt.txt').read_text()
        torch.set_num_threads(2)
        assert self.renderer.device.index < torch.cuda.device_count()
        assert all((PROJECT / 'ckpt' / name).is_file() for name in (
            'sam3d/pipeline_ssonly.yaml', 'sam3d/ss_generator.yaml', 'sam3d/ss_generator.ckpt',
            'sam3d/ss_decoder.yaml', 'sam3d/ss_decoder.ckpt', 'dinov2_vitl14_reg4_pretrain.pth',
            'u2net.onnx', 'isnet-general-use.onnx'))
        assert (PROJECT / 'ckpt/dinov3').is_dir()
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
            config = dict(version=4, source=str(job.source.resolve()), sourceSha256=sha256(job.source),
                          sceneId=job.scene_id, objectId=job.object_id, seed=job.seed,
                          views=dict(anglesRadians=SIX_ANGLES.tolist(), size=VIEW_SIZE,
                                     filterIndices=[0, 1, 2, 3], rodinIndices=list(RODIN_VIEW_INDICES)),
                          fal=dict(model=FAL_MODEL, settings=FAL_SETTINGS, prompt=self.prompt,
                                   retry=dict(model=NANO_MODEL, settings=dict(NANO_SETTINGS, seed=job.seed),
                                              minimumRetainedFraction=MIN_FOREGROUND_RETAINED_FRACTION)),
                          rodin=RODIN_SETTINGS, alignment=ALIGNMENT_SETTINGS,
                          maxRetry=self.max_retry, retryPolicy='alternate Rodin-only and FAL+Rodin')
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
            while True:
                self.generate(task, self.api)
                self.postprocess(task)
                if task.retry_phase is None:
                    break
                if task.retry_phase == 1:
                    self.refine(task, self.api)
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
        self.phase(task, 'SAM3D five-view orientation and Gaussian six-view render')
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
                    write_rgb(gaussian_image, join_views(self.renderer.render(data, frame, SIX_ANGLES, VIEW_SIZE)))
                assert read_rgb(gaussian_image).shape == (3 * VIEW_SIZE, 2 * VIEW_SIZE, 3)
        task.prepared = data, frame

    def refine(self, task: Task, api: GenerationAPI) -> None:
        work, timing = task.reference_work, task.timing
        work.mkdir(parents=True, exist_ok=True)
        for name in ('canonical.json', 'gaussian.png'):
            shutil.copyfile(task.work / name, work / name)
        if task.attempt_index >= 2:
            with timing.stage('retry_read_and_prefilter'):
                data, _ = covariance_prefilter(read_ply(task.job.source))
                task.prepared = data, json.loads((work / 'canonical.json').read_text())
        assert task.prepared is not None
        data, frame = task.prepared
        filtered_path = work / 'gaussian_filter.ply'
        if filtered_path.exists():
            task.image_only = (work / 'image-only.json').is_file()
            with timing.stage('fal_refine', cached=True):
                state = json.loads((work / 'fal-request.json').read_text())
                assert state['status'] == 'completed' and state['outputSha256'] == sha256(work / 'fal_refine.png')
            with timing.stage('foreground_filter', cached=True):
                assert task.image_only or len(read_ply(filtered_path)['xyz']) > 0
        else:
            selection = dict(beforeCount=len(data['xyz']), minimumRetainedFraction=MIN_FOREGROUND_RETAINED_FRACTION,
                             selectedModel=None, attempts=[])
            seed = (task.job.seed + task.attempt_index // 2) % 65536
            attempts = ((FAL_MODEL, 'fal_gpt', None), (NANO_MODEL, 'fal_nano', seed),
                        (NANO_MODEL, 'fal_nano_2', (seed + 1) % 65536))
            for index, (model, directory, seed) in enumerate(attempts):
                attempt = work / directory
                attempt.mkdir(exist_ok=True)
                self.phase(task, f'{model}: six-view refinement')
                with timing.stage('fal_refine' if seed is None else f'fal_nano_retry_{index}', cached=(attempt / 'fal_refine.png').exists()):
                    fal_image = api.refine(work / 'gaussian.png', self.prompt, attempt, nano_seed=seed)
                    views = split_views(read_rgb(fal_image))
                with self.segmentation.use(timing):
                    with timing.stage('foreground_filter' if seed is None else f'foreground_filter_nano_{index}'):
                        self.phase(task, 'U2Net + ISNet on the first four views only; single-view background veto')
                        if self.segmenter is None:
                            self.segmenter = ForegroundSegmenter(PROJECT / 'ckpt/u2net.onnx', PROJECT / 'ckpt/isnet-general-use.onnx')
                        alpha, _, _ = self.segmenter.multiview(views[:4])
                        filtered = filter_foreground(data, frame, alpha)
                retained = len(filtered['xyz']) / len(data['xyz'])
                accepted = retained >= MIN_FOREGROUND_RETAINED_FRACTION
                task.image_only = not accepted and index == 2
                state = json.loads((attempt / 'fal-request.json').read_text())
                selection['attempts'].append(dict(model=model, seed=seed, requestId=state['job']['request_id'],
                                                  afterCount=len(filtered['xyz']), retainedFraction=retained,
                                                  status='selected' if accepted else 'rejected'))
                if accepted:
                    selection['selectedModel'] = model
                if task.image_only:
                    selection['selectedModel'] = FAL_MODEL
                    selection['attempts'][0]['status'] = 'selected_images_only'
                write_json(work / 'foreground_filter.json', selection)
                self.phase(task, f'{model}: retained {len(filtered["xyz"])}/{len(data["xyz"])} Gaussians '
                                f'({retained:.2%}); {"accepted" if accepted else "below 30% retention"}')
                if not accepted and not task.image_only:
                    self.phase(task, f'More than 70% removed; starting Nano Banana attempt {index + 1}/2 using the original input')
                    continue
                if task.image_only:
                    self.phase(task, 'Both Nano attempts retained fewer than 30%; Rodin will use the original GPT images only, without point-cloud conditioning')
                    write_json(work / 'image-only.json', dict(reason='Both Nano attempts retained fewer than 30%', nanoAttempts=2,
                               imageModel=FAL_MODEL,
                               retainedCount=len(filtered['xyz']), beforeCount=len(data['xyz']),
                               minimumRetainedFraction=MIN_FOREGROUND_RETAINED_FRACTION))
                selected_attempt = work / 'fal_gpt' if task.image_only else attempt
                shutil.copyfile(selected_attempt / 'fal_refine.png', work / 'fal_refine.png')
                shutil.copyfile(selected_attempt / 'fal-request.json', work / 'fal-request.json')
                temporary = work / '.gaussian_filter.ply'
                write_ply(task.job.source, filtered['records'], temporary)
                temporary.replace(filtered_path)
                break
            assert filtered_path.exists()
        canonical_to_world = np.asarray(frame['canonicalToWorld']).reshape(4, 4, order='F')
        filtered_image = work / 'gaussian_filter.png'
        condition_path = work / 'condition.npz'
        with self.gpu.use(timing):
            with timing.stage('filtered_gaussian_render', cached=filtered_image.exists()):
                if not filtered_image.exists():
                    self.phase(task, 'Filtered Gaussian six-view render with the original cameras')
                    filtered = read_ply(filtered_path)
                    views = (np.full((6, VIEW_SIZE, VIEW_SIZE, 3), 255, np.uint8) if len(filtered['xyz']) == 0
                             else self.renderer.render(filtered, frame, SIX_ANGLES, VIEW_SIZE))
                    write_rgb(filtered_image, join_views(views))
            with timing.stage('sampling_fps_normalization', cached=condition_path.exists(), skipped=task.image_only):
                if not task.image_only and not condition_path.exists():
                    self.phase(task, 'Gaussian volume sampling, FPS 4096, independent normalization to [-0.98, 0.98]')
                    canonical_points = self.sampler.sample(read_ply(filtered_path), canonical_to_world)
                    condition, normalization = normalize_points(canonical_points)
                    temporary = work / '.condition.npz'
                    np.savez(temporary, canonical=canonical_points, points=condition, normalization=normalization)
                    temporary.replace(condition_path)
        task.prepared = None

    def generate(self, task: Task, api: GenerationAPI) -> None:
        work, timing = task.generation_work, task.timing
        work.mkdir(parents=True, exist_ok=True)
        reference = task.reference_work
        self.phase(task, 'Rodin Gen-2.5-Medium / medium / ' + ('images only' if task.image_only else 'point-cloud weight 0.3'))
        with timing.stage('rodin_generate', cached=(work / 'rodin.glb').exists()):
            condition = None
            if not task.image_only:
                with np.load(reference / 'condition.npz') as saved:
                    condition = saved['points']
            views = split_views(read_rgb(reference / 'fal_refine.png'))
            raw_mesh = api.rodin([views[index] for index in RODIN_VIEW_INDICES], condition,
                                 (task.job.seed + task.attempt_index) % 65536, work)
            validate_glb(raw_mesh)

    def postprocess(self, task: Task, alignment_worker=None) -> None:
        work, timing = task.work, task.timing
        candidate = task.generation_work
        self.phase(task, f'Bbox / conditional ICP / joint DINO registration and scoring, attempt {task.attempt_index}')
        if alignment_worker is None:
            with self.gpu.use(timing), self.segmentation.use(timing):
                with timing.stage('registration_and_scoring', cached=(candidate / 'alignment.json').exists()):
                    if self.matcher is None:
                        self.matcher = DinoMatcher(PROJECT / 'ckpt/dinov3', self.renderer.device)
                    if self.segmenter is None:
                        self.segmenter = ForegroundSegmenter(PROJECT / 'ckpt/u2net.onnx', PROJECT / 'ckpt/isnet-general-use.onnx')
                    result = align_generation(candidate / 'rodin.glb', task.job.source, task.reference_work,
                                              candidate, self.matcher, self.sampler, self.segmenter, self.renderer.device)
        else:
            pool, device = alignment_worker
            with timing.stage('registration_and_scoring', cached=(candidate / 'alignment.json').exists()):
                result = pool.submit(register_generation, candidate / 'rodin.glb', task.job.source,
                                     task.reference_work, candidate, device).result()
        task.candidates.append(dict(attempt=task.attempt_index, reference=task.attempt_index // 2,
                                    seed=(task.job.seed + task.attempt_index) % 65536,
                                    score=result['score'], stage=result['finalStage'], status=result['status'],
                                    matrix=result['matrix'], stages=result['stages']))
        best = max(task.candidates, key=lambda c: -1 if c['score'] is None else c['score'])
        accepted = best['score'] is not None and best['score'] >= MINIMUM_SCORE
        retry = not accepted and task.attempt_index < self.max_retry
        summary = dict(status='retrying' if retry else 'accepted' if accepted else 'below_threshold',
                       minimumScore=MINIMUM_SCORE, maxRetry=self.max_retry, selectedAttempt=best['attempt'],
                       selectedStage=best['stage'], score=best['score'], attempts=task.candidates)
        write_json(work / 'alignment.json', summary)
        if retry:
            task.attempt_index += 1
            task.retry_phase = 2 if task.attempt_index % 2 else 1
            self.phase(task, f'Best score {best["score"]}; retry {task.attempt_index}/{self.max_retry}: '
                            + ('Rodin only' if task.retry_phase == 2 else 'FAL + Rodin'))
            return
        task.retry_phase = None
        selected_reference = work / 'refinements' / f'{best["reference"]:03d}'
        selected_generation = work / 'attempts' / f'{best["attempt"]:03d}'
        with timing.stage('write_transform'):
            matrix = np.asarray(best['matrix'], np.float64).reshape(4, 4, order='F')
            transform = dict(matrixOrder='column-major', rodinToWorld=dict(matrix=best['matrix']),
                             alignment=summary['status'], alignmentScore=best['score'],
                             selectedAttempt=best['attempt'], finalStage=best['stage'], alignmentReport=summary)
            if (selected_reference / 'image-only.json').exists():
                transform['imageOnly'] = True
            write_json(work / 'transform.json', transform)
            for name in ('fal_refine.png', 'fal-request.json', 'gaussian_filter.ply', 'gaussian_filter.png',
                         'foreground_filter.json'):
                shutil.copyfile(selected_reference / name, work / name)
            sampling_file = 'image-only.json' if (selected_reference / 'image-only.json').exists() else 'condition.npz'
            shutil.copyfile(selected_reference / sampling_file, work / sampling_file)
            shutil.copyfile(selected_generation / 'rodin.glb', work / 'rodin.glb')
            shutil.copyfile(selected_generation / 'rodin-request.json', work / 'rodin-request.json')
        with self.gpu.use(timing):
            with timing.stage('mesh_render'):
                frame = json.loads((work / 'canonical.json').read_text())
                write_rgb(work / 'rodin.png', join_views(self.mesh_renderer.render(work / 'rodin.glb', matrix, frame)))
        self.phase(task, 'Export selected model, transform, scores and references')
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
