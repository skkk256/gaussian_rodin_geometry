from pathlib import Path

import numpy as np
import torch
import trimesh

from render import camera_matrices
from views import SIX_ANGLES, VIEW_SIZE


class MeshRenderer:
    def __init__(self, device: str):
        self.device = torch.device(device)
        assert self.device.type == 'cuda' and self.device.index is not None
        self.context = None
        self.warmed = False

    def render(self, path: Path, matrix: np.ndarray, frame: dict) -> np.ndarray:
        """Rodin-to-world float64 [4,4], camera frame -> RGB uint8 [6,512,512,3]."""
        assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
        mesh = trimesh.load_scene(path, process=False).to_mesh()
        mesh.apply_transform(matrix)
        points = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        matrices, intrinsics = camera_matrices(frame['obb'], frame['normalizationScale'], SIX_ANGLES, VIEW_SIZE)
        near, far = .01 / frame['normalizationScale'], 20 / frame['normalizationScale']
        camera = np.einsum('vij,nj->vni', matrices[:, :3, :3], points) + matrices[:, None, :3, 3]
        clip = np.empty((6, len(points), 4), np.float32)
        clip[:, :, 0] = 2 * intrinsics[:, 0, 0, None] * camera[:, :, 0] / VIEW_SIZE
        clip[:, :, 1] = -2 * intrinsics[:, 1, 1, None] * camera[:, :, 1] / VIEW_SIZE
        clip[:, :, 2] = (far + near) / (far - near) * camera[:, :, 2] - 2 * far * near / (far - near)
        clip[:, :, 3] = camera[:, :, 2]
        camera_normals = np.einsum('vij,nj->vni', matrices[:, :3, :3], normals)
        with torch.cuda.device(self.device):
            import nvdiffrast.torch as dr

            if self.context is None:
                self.context = dr.RasterizeCudaContext(device=self.device)
            positions = torch.as_tensor(clip, device=self.device).contiguous()
            faces = torch.as_tensor(np.asarray(mesh.faces, dtype=np.int32), device=self.device).contiguous()
            normal = torch.as_tensor(camera_normals, device=self.device).contiguous()
            assert positions.device == faces.device == normal.device == self.device
            if not self.warmed:
                torch.cuda.empty_cache()
                raster, _ = dr.rasterize(self.context, positions, faces, (VIEW_SIZE, VIEW_SIZE))
                warm, _ = dr.interpolate(normal, raster, faces)
                dr.antialias(warm, raster, positions, faces)
                torch.cuda.current_stream(self.device).synchronize()
                del raster, warm
                torch.cuda.empty_cache()
                self.warmed = True
            raster, _ = dr.rasterize(self.context, positions, faces, (VIEW_SIZE, VIEW_SIZE))
            interpolated, _ = dr.interpolate(normal, raster, faces)
            interpolated = torch.nn.functional.normalize(interpolated, dim=-1)
            light = torch.as_tensor([-.35, -.45, -.82], dtype=torch.float32, device=self.device)
            shade = .35 + .65 * torch.abs((interpolated * light).sum(-1, keepdim=True))
            color = shade * torch.as_tensor([.65, .72, .79], device=self.device)
            color = torch.where((raster[..., 3] > 0)[..., None], color, torch.ones_like(color))
            color = dr.antialias(color, raster, positions, faces)
            return (color.flip(1).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
