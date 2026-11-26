from pathlib import Path
import os
import signal
import time

import torch

from align.generation import align_generation
from align.matching import DinoMatcher
from conditions import PointSampler
from foreground import ForegroundSegmenter

PROJECT = Path(__file__).resolve().parents[1]
_matcher = _sampler = _segmenter = None


def register_generation(mesh: Path, source: Path, work: Path, folder: Path, device: str) -> dict:
    global _matcher, _sampler, _segmenter
    target = torch.device(device)
    assert target.type == 'cuda' and target.index is not None
    if _matcher is None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        torch.set_num_threads(2)
        _matcher = DinoMatcher(PROJECT / 'ckpt/dinov3', target)
        _sampler = PointSampler(device)
        _segmenter = ForegroundSegmenter(PROJECT / 'ckpt/u2net.onnx', PROJECT / 'ckpt/isnet-general-use.onnx')
    assert _matcher.device == _sampler.device == target
    started = time.perf_counter()
    print(f'ALIGN WORKER pid={os.getpid()} device={device} start {folder}', flush=True)
    with torch.cuda.device(target):
        report = align_generation(mesh, source, work, folder, _matcher, _sampler, _segmenter, target)
    print(f'ALIGN WORKER pid={os.getpid()} device={device} end {folder}: {time.perf_counter()-started:.2f}s', flush=True)
    return report
