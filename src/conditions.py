import numpy as np
from scipy import ndimage
import torch

from gaussian import quaternion_matrix
from render import FOUR_ANGLES, camera_matrices


def filter_foreground(data: dict, frame: dict, alpha: np.ndarray) -> dict:
    """Gaussian float64 arrays [N,...], alpha float32 [4,1024,1024] -> selected Gaussian arrays [M,...]."""
    assert alpha.shape == (4, 1024, 1024)
    masks = np.stack([ndimage.binary_dilation(a >= .25, iterations=4) for a in alpha])
    matrices, intrinsics = camera_matrices(frame['obb'], frame['normalizationScale'], FOUR_ANGLES, 1024)
    keep = np.ones(len(data['xyz']), dtype=bool)
    for matrix, k, mask in zip(matrices, intrinsics, masks):
        camera = data['xyz'] @ matrix[:3, :3].T + matrix[:3, 3]
        uv = np.rint(camera[:, :2] / np.maximum(camera[:, 2, None], 1e-5)
                     * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]] - .5).astype(int)
        valid = (camera[:, 2] > 1e-5) & np.all((uv >= 0) & (uv < 1024), axis=1)
        keep[valid] &= mask[uv[valid, 1], uv[valid, 0]]
    return {key: value[keep] for key, value in data.items()}


class PointSampler:
    def __init__(self, device: str):
        self.device = torch.device(device)
        assert self.device.type == 'cuda' and self.device.index is not None
        self.warmed = False

    def sample(self, data: dict, canonical_to_world: np.ndarray) -> np.ndarray:
        """Gaussian float64 arrays [N,...], canonical_to_world float64 [4,4] -> canonical XYZ float32 [4096,3]."""
        eligible = np.flatnonzero(data['opacity'] > .2)
        assert len(eligible) > 0, 'No filtered Gaussian has opacity > 0.2'
        rng = np.random.default_rng(42)
        ids = rng.choice(eligible, 65536, replace=True)
        directions = rng.normal(size=(len(ids), 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.random(len(ids)) ** (1 / 3) * np.sqrt(2 * np.log(data['opacity'][ids] / .2))
        rotation = quaternion_matrix(data['rotation'][ids])
        displacement = np.einsum('nij,nj->ni', rotation, data['scale'][ids] * directions * radii[:, None])
        world = data['xyz'][ids] + displacement
        inverse = np.linalg.inv(canonical_to_world)
        candidates = (world @ inverse[:3, :3].T + inverse[:3, 3]).astype(np.float32)
        candidates = candidates[np.all(np.abs(candidates) < .98, axis=1)]
        assert len(candidates) >= 4096 and np.isfinite(candidates).all()
        with torch.cuda.device(self.device):
            from pytorch3d.ops import sample_farthest_points

            points = torch.as_tensor(candidates, device=self.device)[None]
            if not self.warmed:
                torch.cuda.empty_cache()
                sample_farthest_points(points, K=4096, random_start_point=False)
                torch.cuda.current_stream(self.device).synchronize()
                torch.cuda.empty_cache()
                self.warmed = True
            samples, indices = sample_farthest_points(points, K=4096, random_start_point=False)
            samples = samples[0].cpu().numpy()
            assert len(np.unique(indices[0].cpu().numpy())) == 4096
        return samples


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """canonical XYZ float32/float64 [4096,3] -> condition XYZ float32 [4096,3], canonical-to-condition float64 [4,4]."""
    assert points.shape == (4096, 3) and np.isfinite(points).all()
    points = points.astype(np.float64)
    center = (points.min(0) + points.max(0)) / 2
    extent = float(np.ptp(points, axis=0).max())
    assert extent > 0
    matrix = np.eye(4)
    matrix[:3, :3] *= 1.96 / extent
    matrix[:3, 3] = -center * 1.96 / extent
    normalized = (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
    assert np.allclose(normalized.min(0) + normalized.max(0), 0, atol=2e-7)
    assert np.isclose(np.ptp(normalized, axis=0).max(), 1.96, atol=2e-7)
    return normalized, matrix
