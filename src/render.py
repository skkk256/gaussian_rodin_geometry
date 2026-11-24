from math import isqrt

import numpy as np
import torch


def camera_matrices(obb: dict, scale: float, angles: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth/elevation angles float64 [V,2] -> OpenCV world-to-camera float32 [V,4,4], intrinsics float32 [V,3,3]."""
    axes = np.asarray(obb['axes'], dtype=np.float64)
    center = np.asarray(obb['center'], dtype=np.float64)
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-6) and np.linalg.det(axes) > 0
    assert angles.ndim == 2 and angles.shape[1] == 2
    azimuth, elevation = angles.T
    distance = 2.5 * np.cos(elevation)
    eyes = np.stack([distance * np.sin(azimuth), 2.5 * np.sin(elevation), distance * np.cos(azimuth)], axis=1)
    forward = -eyes / np.linalg.norm(eyes, axis=1, keepdims=True)
    right = np.stack([np.cos(azimuth), np.zeros_like(azimuth), -np.sin(azimuth)], axis=1)
    down = -np.cross(right, forward)
    rotations = np.stack((right, down, forward), axis=1) @ axes
    world_eyes = center + eyes @ axes / scale
    matrices = np.tile(np.eye(4), (len(angles), 1, 1))
    matrices[:, :3, :3] = rotations
    matrices[:, :3, 3] = -np.einsum('nij,nj->ni', rotations, world_eyes)
    focal = resolution * 2.5 / (2 * .58)
    intrinsics = np.tile([[focal, 0, resolution / 2], [0, focal, resolution / 2], [0, 0, 1]], (len(angles), 1, 1))
    return matrices.astype(np.float32), intrinsics.astype(np.float32)


class GaussianRenderer:
    def __init__(self, device: str):
        self.device = torch.device(device)
        assert self.device.type == 'cuda' and self.device.index is not None
        self.warmed = set()

    def render(self, data: dict, frame: dict, angles: np.ndarray, resolution: int) -> np.ndarray:
        """Gaussian float64 arrays [N,...], angles float64 [V,2] -> RGB uint8 [V,resolution,resolution,3]."""
        matrices, intrinsics = camera_matrices(frame['obb'], frame['normalizationScale'], angles, resolution)
        records = data['records']
        dc = np.column_stack([records[f'f_dc_{i}'] for i in range(3)])[:, None, :]
        rest_names = sorted((name for name in records.dtype.names if name.startswith('f_rest_')),
                            key=lambda name: int(name.removeprefix('f_rest_')))
        assert len(rest_names) in (0, 45)
        coefficients = 1 + len(rest_names) // 3
        degree = isqrt(coefficients) - 1
        rest = np.asarray([records[name] for name in rest_names], dtype=np.float32).T
        rest = rest.reshape(len(records), 3, coefficients - 1).transpose(0, 2, 1)
        colors = np.concatenate((dc, rest), axis=1)
        assert np.all(np.linalg.norm(data['rotation'], axis=1) > 1e-12)
        with torch.cuda.device(self.device):
            from gsplat import rasterization

            arguments = dict(
                means=torch.as_tensor(data['xyz'], dtype=torch.float32, device=self.device),
                quats=torch.as_tensor(data['rotation'], dtype=torch.float32, device=self.device),
                scales=torch.as_tensor(data['scale'], dtype=torch.float32, device=self.device),
                opacities=torch.as_tensor(data['opacity'], dtype=torch.float32, device=self.device),
                colors=torch.as_tensor(colors, dtype=torch.float32, device=self.device),
                viewmats=torch.as_tensor(matrices, device=self.device),
                Ks=torch.as_tensor(intrinsics, device=self.device), width=resolution, height=resolution,
                near_plane=.01 / frame['normalizationScale'], far_plane=20 / frame['normalizationScale'],
                sh_degree=degree, packed=False, camera_model='pinhole',
                backgrounds=torch.ones((len(angles), 3), device=self.device))
            key = (len(angles), resolution, degree)
            if key not in self.warmed:
                torch.cuda.empty_cache()
                rasterization(**arguments)
                torch.cuda.current_stream(self.device).synchronize()
                torch.cuda.empty_cache()
                self.warmed.add(key)
            rgb, alpha, _ = rasterization(**arguments)
            assert torch.isfinite(rgb).all() and torch.isfinite(alpha).all()
            return rgb.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
