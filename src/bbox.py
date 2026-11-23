from pathlib import Path

import numpy as np
import trimesh


def register_bbox(raw_mesh: Path, target_world: np.ndarray, inverse: np.ndarray,
                  obb: dict) -> np.ndarray:
    """target_world float64 [4096,3], inverse float64 [4,4], obb axes/center float64 [3,3]/[3] -> Rodin-to-world float64 [4,4]."""
    assert target_world.shape == (4096, 3) and inverse.shape == (4, 4)
    assert np.isfinite(target_world).all() and np.isfinite(inverse).all()
    mesh = trimesh.load_scene(raw_mesh, process=False).to_mesh()
    assert len(mesh.faces) > 0 and np.isfinite(mesh.vertices).all() and mesh.area > 0
    raw_points, _ = trimesh.sample.sample_surface(mesh, 8000, seed=0)
    world_to_bbox = np.eye(4)
    world_to_bbox[:3, :3] = np.asarray(obb['axes'])
    world_to_bbox[:3, 3] = -world_to_bbox[:3, :3] @ np.asarray(obb['center'])
    raw_to_bbox = world_to_bbox @ inverse
    source = raw_points @ raw_to_bbox[:3, :3].T + raw_to_bbox[:3, 3]
    target = target_world @ world_to_bbox[:3, :3].T + world_to_bbox[:3, 3]
    source_bounds = np.percentile(source, [0.25, 99.75], axis=0)
    target_bounds = np.percentile(target, [0.25, 99.75], axis=0)
    source_extent = np.ptp(source_bounds, axis=0).max()
    target_extent = np.ptp(target_bounds, axis=0).max()
    assert source_extent > 0 and target_extent > 0
    scale = target_extent / source_extent
    correction = np.eye(4)
    correction[:3, :3] *= scale
    correction[:3, 3] = target_bounds.mean(0) - scale * source_bounds.mean(0)
    world_correction = np.linalg.solve(world_to_bbox, correction @ world_to_bbox)
    assert np.allclose(world_correction[:3, :3], np.eye(3) * scale, atol=1e-10)
    return world_correction @ inverse
