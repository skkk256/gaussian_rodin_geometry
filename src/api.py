import base64
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from io_utils import png_bytes, sha256, validate_glb, write_json

FAL_MODEL = 'openai/gpt-image-2.6/sunburst/edit'
FAL_SETTINGS = dict(num_images=1, image_size=dict(width=2048, height=2048), quality='high',
                    reasoning_effort='high', output_format='png', background='opaque')
NANO_MODEL = 'google/nano-banana-2.1/edit'
NANO_SETTINGS = dict(num_images=1, aspect_ratio='1:1', resolution='2K', output_format='png', thinking_level='high')
RODIN_API = 'https://api.hyper3d.com/api/v2'
RODIN_SETTINGS = dict(tier='Gen-2.5-Medium', quality='medium', mesh_mode='Raw',
                      geometry_file_format='glb', material='None', use_original_alpha='true',
                      is_symmetric='unknown', image_label=['F', 'L', 'B', 'R'],
                      pcd_condition_cfg='true', pcd_condition_weight='0.3')


class GenerationAPI:
    def __init__(self):
        self.fal_key = os.environ['FAL_API_KEY']
        self.rodin_key = os.environ['RODIN_API_KEY']
        assert self.fal_key and self.rodin_key
        self.session = requests.Session()
        self.session.trust_env = False
        retries = Retry(total=5, backoff_factor=1, allowed_methods={'GET'}, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def refine(self, image: Path, prompt: str, work: Path, *, nano_seed: int | None = None) -> Path:
        model = FAL_MODEL if nano_seed is None else NANO_MODEL
        settings = FAL_SETTINGS if nano_seed is None else dict(NANO_SETTINGS, seed=nano_seed)
        assert nano_seed is None or 0 <= nano_seed < 65536
        request = dict(model=model, settings=dict(settings, prompt=prompt), inputSha256=sha256(image))
        journal = work / 'fal-request.json'
        target = work / 'fal_refine.png'
        if journal.exists():
            state = json.loads(journal.read_text())
            assert state['request'] == request, 'Fal inputs changed; use a new work directory'
            assert state['status'] in ('submitted', 'completed'), 'Uncertain Fal submission; inspect fal-request.json before sending another paid request'
        else:
            state = dict(status='submitting', request=request)
            write_json(journal, state)
            encoded = base64.b64encode(image.read_bytes()).decode('ascii')
            response = self.session.post('https://queue.fal.run/' + model,
                headers={'Authorization': 'Key ' + self.fal_key},
                json=dict(request['settings'], image_urls=['data:image/png;base64,' + encoded]), timeout=180)
            response.raise_for_status()
            job = response.json()
            assert job['request_id'] and job['status_url'] and job['response_url']
            state.update(status='submitted', job=job)
            write_json(journal, state)
        if target.exists():
            assert state['status'] == 'completed' and state['outputSha256'] == sha256(target)
            return target
        job = state['job']
        deadline = time.monotonic() + 3600
        while True:
            assert time.monotonic() < deadline, 'Fal timeout; rerun the same command to resume'
            response = self.session.get(job['status_url'], headers={'Authorization': 'Key ' + self.fal_key}, timeout=60)
            response.raise_for_status()
            status = response.json()['status']
            if status == 'COMPLETED':
                break
            assert status in ('IN_QUEUE', 'IN_PROGRESS'), status
            time.sleep(5)
        response = self.session.get(job['response_url'], headers={'Authorization': 'Key ' + self.fal_key}, timeout=60)
        response.raise_for_status()
        result = response.json()
        assert len(result['images']) == 1
        response = self.session.get(result['images'][0]['url'], timeout=300)
        response.raise_for_status()
        temporary = target.with_suffix('.downloading')
        temporary.write_bytes(response.content)
        state.update(status='completed', outputSha256=sha256(temporary))
        write_json(journal, state)
        temporary.replace(target)
        return target

    def rodin(self, images: list[np.ndarray], points: np.ndarray | None, seed: int, work: Path) -> Path:
        """Four RGB uint8 [1024,1024,3] images, condition XYZ float32 [4096,3] or None -> raw GLB path."""
        assert len(images) == 4
        assert points is None or (points.shape == (4096, 3) and np.isfinite(points).all())
        assert 0 <= seed < 65536
        encoded = [png_bytes(image) for image in images]
        condition = None if points is None else json.dumps(points.tolist(), separators=(',', ':'), allow_nan=False)
        settings = dict(RODIN_SETTINGS, seed=seed)
        if points is None:
            del settings['pcd_condition_cfg'], settings['pcd_condition_weight']
        request = dict(api=RODIN_API, settings=settings,
                       imageSha256=[hashlib.sha256(body).hexdigest() for body in encoded],
                       conditionSha256=None if condition is None else hashlib.sha256(condition.encode()).hexdigest())
        headers = {'Authorization': 'Bearer ' + self.rodin_key}
        journal = work / 'rodin-request.json'
        target = work / 'rodin.glb'
        if journal.exists():
            state = json.loads(journal.read_text())
            assert state['request'] == request, 'Rodin inputs changed; use a new work directory'
            assert state['status'] in ('submitted', 'completed'), 'Uncertain Rodin submission; inspect rodin-request.json before sending another paid request'
        else:
            state = dict(status='submitting', request=request)
            write_json(journal, state)
            form = [(key, json.dumps(value) if isinstance(value, list) else str(value)) for key, value in settings.items()]
            if condition is not None:
                form.append(('pcd_condition', condition))
            response = self.session.post(RODIN_API + '/rodin', headers=headers, data=form,
                files=[('images', (f'image-{i + 1}.png', body, 'image/png')) for i, body in enumerate(encoded)], timeout=600)
            response.raise_for_status()
            job = response.json()
            assert job['uuid'] and job['jobs']['subscription_key']
            state.update(status='submitted', job=job)
            write_json(journal, state)
        if target.exists():
            assert state['status'] == 'completed' and state['outputSha256'] == sha256(target)
            validate_glb(target)
            return target
        job = state['job']
        deadline = time.monotonic() + 5400
        while True:
            assert time.monotonic() < deadline, 'Rodin timeout; rerun the same command to resume'
            response = self.session.post(RODIN_API + '/status', headers=headers,
                json={'subscription_key': job['jobs']['subscription_key']}, timeout=180)
            response.raise_for_status()
            statuses = [record['status'] for record in response.json()['jobs']]
            assert statuses and not any(any(word in s.lower() for word in ('failed', 'error', 'cancel')) for s in statuses), statuses
            if all(status == 'Done' for status in statuses):
                break
            time.sleep(5)
        response = self.session.post(RODIN_API + '/download', headers=headers, json={'task_uuid': job['uuid']}, timeout=180)
        response.raise_for_status()
        entry = next(record for record in response.json()['list'] if record['name'].lower().endswith('.glb'))
        response = self.session.get(entry['url'], timeout=300)
        response.raise_for_status()
        temporary = target.with_suffix('.downloading')
        temporary.write_bytes(response.content)
        validate_glb(temporary)
        state.update(status='completed', outputSha256=sha256(temporary))
        write_json(journal, state)
        temporary.replace(target)
        return target
