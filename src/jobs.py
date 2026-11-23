import json
from pathlib import Path

from pipeline import Job


def load_jobs(manifest: Path) -> list[Job]:
    manifest = manifest.resolve(strict=True)
    rows = json.loads(manifest.read_text())
    assert isinstance(rows, list) and rows
    jobs = []
    for row in rows:
        assert set(row) == {'scene_id', 'object_id', 'input', 'seed'}, 'Each manifest row must contain scene_id, object_id, input, seed'
        job = Job((manifest.parent / row['input']).resolve(strict=True), row['scene_id'], row['object_id'], row['seed'])
        job.validate()
        jobs.append(job)
    assert len({(job.scene_id, job.object_id) for job in jobs}) == len(jobs), 'Duplicate scene_id/object_id'
    return jobs
