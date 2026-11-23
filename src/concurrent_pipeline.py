from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import signal
import threading
import time
import traceback

from api import GenerationAPI
from io_utils import write_json
from pipeline import Job, Pipeline, Task


PHASES = ('preprocess_sam3d', 'fal', 'rodin', 'postprocess')


class ConcurrentBatch:
    def __init__(self, pipeline: Pipeline, fal_workers: int, rodin_workers: int, max_inflight: int):
        assert fal_workers > 0 and rodin_workers > 0 and max_inflight > 0
        self.pipeline = pipeline
        self.workers = (1, fal_workers, rodin_workers, 1)
        self.max_inflight = max_inflight
        self.local = threading.local()
        self.stopping = False

    def stop(self, signum, frame) -> None:
        self.stopping = True
        print('Stop requested: no new objects; finishing admitted objects and saving results.', flush=True)

    def execute(self, task: Task, phase: int) -> None:
        name = PHASES[phase]
        print(f'{task.job.scene_id}/{task.job.object_id}: start {name}', flush=True)
        with task.timing.phase(name, task.queued_at):
            if phase in (1, 2) and not hasattr(self.local, 'api'):
                self.local.api = GenerationAPI()
            if phase == 0:
                self.pipeline.prepare(task)
            elif phase == 1:
                self.pipeline.refine(task, self.local.api)
            elif phase == 2:
                self.pipeline.generate(task, self.local.api)
            else:
                self.pipeline.postprocess(task)
        stage = task.timing.attempt['stages'][-1]
        print(f'{task.job.scene_id}/{task.job.object_id}: end {name}: {stage["processingSeconds"] + stage["overheadSeconds"]:.2f}s '
              f'(processing {stage["processingSeconds"]:.2f}s, queue {stage["queueSeconds"]:.2f}s, '
              f'overhead {stage["overheadSeconds"]:.2f}s)', flush=True)

    def run(self, jobs: list[Job], output: Path, work: Path) -> None:
        started = time.perf_counter()
        report = dict(startedAt=datetime.now(timezone.utc).isoformat(), status='running',
                      workers=dict(zip(PHASES, self.workers)), maxInflight=self.max_inflight,
                      completed=[], skipped=[], failed=[], remaining=len(jobs), totalSeconds=0.)
        report_path = work.resolve() / 'batch-timing.json'
        pending, index = {}, 0
        previous_handler = signal.signal(signal.SIGINT, self.stop)
        print(f'Pipeline workers: {report["workers"]}; max in-flight objects: {self.max_inflight}', flush=True)
        try:
            with ExitStack() as stack:
                pools = [stack.enter_context(ThreadPoolExecutor(max_workers=count, thread_name_prefix=name))
                         for name, count in zip(PHASES, self.workers)]
                while pending or (index < len(jobs) and not self.stopping):
                    while not self.stopping and index < len(jobs) and len(pending) < self.max_inflight:
                        job = jobs[index]
                        index += 1
                        key = f'{job.scene_id}/{job.object_id}'
                        try:
                            task = self.pipeline.begin(job, output, work, flows=True)
                        except Exception:
                            report['failed'].append(key)
                            traceback.print_exc()
                            continue
                        if task is None:
                            report['skipped'].append(key)
                            continue
                        task.queued_at = time.perf_counter()
                        pending[pools[0].submit(self.execute, task, 0)] = task, 0
                    if not pending:
                        break
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        task, phase = pending.pop(future)
                        key = f'{task.job.scene_id}/{task.job.object_id}'
                        try:
                            future.result()
                        except Exception:
                            self.pipeline.fail(task)
                            report['totalSeconds'] += task.timing.attempt['totalSeconds']
                            task.lock.close()
                            report['failed'].append(key)
                            traceback.print_exc()
                            continue
                        if phase + 1 < len(PHASES):
                            task.queued_at = task.timing.last_checkpoint
                            pending[pools[phase + 1].submit(self.execute, task, phase + 1)] = task, phase + 1
                        else:
                            try:
                                self.pipeline.complete(task)
                                report['totalSeconds'] += task.timing.attempt['totalSeconds']
                                report['completed'].append(key)
                            finally:
                                task.lock.close()
                    report.update(wallSeconds=time.perf_counter() - started, remaining=len(jobs) - index,
                                  active=len(pending))
                    write_json(report_path, report)
            report.update(status='stopped' if self.stopping else 'failed' if report['failed'] else 'completed',
                          finishedAt=datetime.now(timezone.utc).isoformat(), wallSeconds=time.perf_counter() - started,
                          remaining=len(jobs) - index, active=0)
            write_json(report_path, report)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
        print(f'Batch {report["status"]}: {len(report["completed"])} completed, {len(report["skipped"])} skipped, '
              f'{len(report["failed"])} failed; processing time {report["totalSeconds"]:.1f}s; '
              f'wall time {report["wallSeconds"]:.1f}s', flush=True)
        assert not report['failed'], f'Failed objects: {report["failed"]}; rerun the same command to resume saved requests'
