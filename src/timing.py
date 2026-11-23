from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from io_utils import write_json


def processing_seconds(stages: list[dict]) -> float:
    return sum(stage['processingSeconds'] + stage['overheadSeconds'] if 'details' in stage else stage['seconds']
               for stage in stages if 'kind' not in stage or stage['kind'] != 'queue')


class Timing:
    def __init__(self, path: Path, started: float, initialization_seconds: float, *, flows: bool = False):
        self.path, self.started = path, started
        self.data = json.loads(path.read_text()) if path.exists() else dict(version=1, attempts=[])
        for attempt in self.data['attempts']:
            attempt['totalSeconds'] = attempt['overheadSeconds'] + processing_seconds(attempt['stages'])
            if attempt['status'] == 'running':
                attempt['status'] = 'interrupted'
                for stage in attempt['stages']:
                    if stage['status'] == 'running':
                        stage['status'] = 'interrupted'
        self.initialization_seconds = initialization_seconds
        self.flows, self.current_phase = flows, None
        self.queue_seconds = 0.
        self.last_checkpoint = started
        self.attempt = dict(startedAt=datetime.fromtimestamp(time.time() - (time.perf_counter() - started) - initialization_seconds, timezone.utc).isoformat(), status='running',
                            totalSeconds=0., overheadSeconds=0., stages=[] if flows else [dict(
                                name='pipeline_initialization', seconds=initialization_seconds,
                                status='cached' if initialization_seconds == 0 else 'completed')])
        self.data['attempts'].append(self.attempt)
        self.save()

    def save(self) -> None:
        self.attempt['wallSeconds'] = time.perf_counter() - self.started + self.initialization_seconds
        self.attempt['queueSeconds'] = self.queue_seconds
        self.attempt['totalSeconds'] = self.attempt['wallSeconds'] - self.queue_seconds
        self.attempt['overheadSeconds'] = self.attempt['totalSeconds'] - processing_seconds(self.attempt['stages'])
        self.data['totalSeconds'] = sum(a['totalSeconds'] for a in self.data['attempts'])
        write_json(self.path, self.data)

    def wait(self, name: str, seconds: float) -> None:
        stages = self.current_phase['details'] if self.current_phase is not None else self.attempt['stages']
        stages.append(dict(name='wait_' + name, seconds=seconds, status='completed', kind='queue'))
        self.queue_seconds += seconds
        self.save()

    @contextmanager
    def phase(self, name: str, queued_at: float):
        assert self.flows and self.current_phase is None
        worker_started = time.perf_counter()
        first = not self.attempt['stages']
        started = self.started - self.initialization_seconds if first else queued_at
        phase = dict(name=name, seconds=0., status='running', queueSeconds=worker_started - queued_at,
                     processingSeconds=0., overheadSeconds=0., details=[])
        self.queue_seconds += phase['queueSeconds']
        if first:
            phase['details'].append(dict(name='pipeline_initialization', seconds=self.initialization_seconds,
                                         status='completed' if self.initialization_seconds else 'cached'))
        self.attempt['stages'].append(phase)
        self.current_phase = phase
        self.save()
        try:
            yield
        except BaseException:
            phase['status'] = 'failed'
            raise
        else:
            phase['status'] = 'completed'
        finally:
            self.last_checkpoint = time.perf_counter()
            phase['seconds'] = self.last_checkpoint - started
            phase['queueSeconds'] += sum(s['seconds'] for s in phase['details'] if 'kind' in s and s['kind'] == 'queue')
            phase['processingSeconds'] = sum(s['seconds'] for s in phase['details'] if 'kind' not in s)
            phase['overheadSeconds'] = phase['seconds'] - phase['queueSeconds'] - phase['processingSeconds']
            self.current_phase = None
            self.save()

    @contextmanager
    def stage(self, name: str, *, cached: bool = False, skipped: bool = False):
        started = time.perf_counter()
        stage = dict(name=name, seconds=0., status='running')
        stages = self.current_phase['details'] if self.current_phase is not None else self.attempt['stages']
        stages.append(stage)
        self.save()
        try:
            yield
        except BaseException:
            stage['status'] = 'failed'
            raise
        else:
            stage['status'] = 'skipped' if skipped else 'cached' if cached else 'completed'
        finally:
            stage['seconds'] = time.perf_counter() - started
            self.save()

    def finish(self, status: str) -> None:
        self.attempt.update(status=status, finishedAt=datetime.now(timezone.utc).isoformat())
        self.save()
