from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch

from gaussian import (compute_semantic_obb, normalize_gaussians, normalized_covariances,
                      rasterize_voxel)
from render import GaussianRenderer
from views import FIVE_ANGLES

EXTENT = .86
CANONICAL_TO_RODIN = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                               [0., -1., 0., 0.], [0., 0., 0., 1.]])


def world_to_obb(obb: dict) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(obb['axes']) * EXTENT / max(obb['size'])
    matrix[:3, 3] = -matrix[:3, :3] @ np.asarray(obb['center'])
    return matrix


def pointmap_view(xyz: np.ndarray, obb: dict, view: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World XYZ float64 [N,3] -> pointmap float32 [512,512,3], mask bool [512,512], world-to-PyTorch3D float64 [4,4]."""
    resolution = 512
    # Keep the original float64 vote-camera computation; renderer matrices are float32.
    eye = np.array([2.5 * np.cos(np.deg2rad(30.)) * np.sin(FIVE_ANGLES[view, 0]),
                    2.5 * np.sin(np.deg2rad(30.)),
                    2.5 * np.cos(np.deg2rad(30.)) * np.cos(FIVE_ANGLES[view, 0])])
    forward = -eye / np.linalg.norm(eye)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    down = -np.cross(right, forward)
    normalized_to_cv = np.eye(4)
    normalized_to_cv[:3, :3] = np.stack((right, down, forward))
    normalized_to_cv[:3, 3] = -normalized_to_cv[:3, :3] @ eye
    matrix = normalized_to_cv @ world_to_obb(obb)
    world_to_p3d = np.diag([-1., -1., 1., 1.]) @ matrix
    points = (matrix @ np.column_stack((xyz, np.ones(len(xyz)))).T).T[:, :3]
    points = points[points[:, 2] > 0]
    focal = resolution * 2.5 / (2. * .58)
    pixels = np.rint(focal * points[:, :2] / points[:, 2, None] + resolution / 2 - .5).astype(int)
    valid = np.all((pixels >= 0) & (pixels < resolution), axis=1)
    pixels, points = pixels[valid], points[valid]
    depth = np.full((resolution, resolution), np.inf, np.float32)
    np.minimum.at(depth, (pixels[:, 1], pixels[:, 0]), points[:, 2])
    visible = np.isfinite(depth)
    assert visible.any()
    nearest = ndimage.distance_transform_edt(~visible, return_distances=False, return_indices=True)
    depth = depth[tuple(nearest)]
    yy, xx = np.indices(depth.shape)
    x = (xx + .5 - resolution / 2) / focal * depth
    y = (yy + .5 - resolution / 2) / focal * depth
    pointmap = np.stack((-x, -y, depth), axis=-1).astype(np.float32)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(ndimage.binary_dilation(visible, iterations=3), iterations=3))
    return pointmap, mask, world_to_p3d


def pose_vote(coords: np.ndarray, canonical_to_p3d: np.ndarray, world_to_p3d: np.ndarray,
              obb: dict, gaussian_grid: np.ndarray) -> tuple[int, float]:
    """coords int64 [N,3], transforms float64 [4,4], gaussian_grid bool [16,16,16] -> quarter turn, tie-break Chamfer."""
    rodin = np.empty_like(coords)
    rodin[:, 0], rodin[:, 1], rodin[:, 2] = coords[:, 0], coords[:, 2], 63 - coords[:, 1]
    grid = np.zeros((16, 16, 16), bool)
    index = rodin // 4
    grid[index[:, 0], index[:, 1], index[:, 2]] = True
    grid = ndimage.binary_fill_holes(grid)
    pose = np.linalg.solve(world_to_p3d, canonical_to_p3d) @ np.linalg.inv(CANONICAL_TO_RODIN @ np.diag([2., 2., 2., 1.]))
    obb_pose = world_to_obb(obb)[:3, :3] @ pose[:3, :3]
    front = world_to_obb(obb)[:3, :3] @ pose[:3, 2]
    assert np.linalg.norm(front[[0, 2]]) > 1e-6
    quarter = int(np.rint(np.arctan2(front[0], front[2]) / (np.pi / 2))) % 4
    u, _, vt = np.linalg.svd(obb_pose)
    rotation = u @ vt
    assert np.linalg.det(rotation) > 0
    source = ((np.argwhere(grid) + .5) / 16 - .5) @ rotation.T
    target = (np.argwhere(gaussian_grid) + .5) / 16 - .5
    scale = np.ptp(target, axis=0).max() / np.ptp(source, axis=0).max()
    source = source * scale + (target.min(0) + target.max(0)) / 2 - scale * (source.min(0) + source.max(0)) / 2
    chamfer = (cKDTree(target).query(source)[0].mean() + cKDTree(source).query(target)[0].mean()) / 2
    return quarter, float(chamfer)


class Orientation:
    def __init__(self, renderer: GaussianRenderer, ckpt: Path):
        # SAM3D imports native modules lazily; initialize them under the owning device.
        with torch.cuda.device(renderer.device):
            from coarse import CoarseInference

            self.model = CoarseInference(ckpt / 'sam3d/pipeline_ssonly.yaml', str(renderer.device))
        self.renderer = renderer

    def prepare(self, data: dict) -> dict:
        """Gaussian float64 arrays [N,...] -> canonical frame with column-major canonical-to-world float64 matrix [16]."""
        obb = compute_semantic_obb(data['xyz'])
        frame = dict(obb=obb, normalizationScale=EXTENT / max(obb['size']))
        images = self.renderer.render(data, frame, FIVE_ANGLES, 512)
        grid = rasterize_voxel(normalize_gaussians(data, obb, EXTENT), normalized_covariances(data, obb, EXTENT))
        assert grid.any()
        votes = []
        for view, image in enumerate(images):
            pointmap, mask, world_to_p3d = pointmap_view(data['xyz'], obb, view)
            coords, canonical_to_p3d = self.model.generate_coarse(image, mask, torch.from_numpy(pointmap), seed=42)
            quarter, chamfer = pose_vote(coords, canonical_to_p3d, world_to_p3d, obb, grid)
            votes.append(dict(view=view, quarter=quarter, chamfer=chamfer))
        counts = [sum(vote['quarter'] == q for vote in votes) for q in range(4)]
        tied = [q for q in range(4) if counts[q] == max(counts)]
        quarter = min(tied, key=lambda q: min(vote['chamfer'] for vote in votes if vote['quarter'] == q))
        angle = quarter * np.pi / 2
        rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
        canonical_to_obb = np.eye(4)
        canonical_to_obb[:3, :3] = rotation.T
        matrix = np.linalg.inv(world_to_obb(obb)) @ canonical_to_obb @ np.diag([.5, .5, .5, 1.])
        return dict(obb=dict(obb, axes=(rotation @ obb['axes']).tolist(), size=(np.abs(rotation) @ obb['size']).tolist()),
                    normalizationScale=frame['normalizationScale'], votedQuarter=quarter, votes=votes,
                    canonicalToWorld=matrix.flatten(order='F').tolist())
