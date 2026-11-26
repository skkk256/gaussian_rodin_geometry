"""Six Gaussian reference views with shared mesh cameras in normalized OBB coordinates."""
import numpy as np

from render import camera_matrices
from views import VIEW_SIZE

FIT = np.arange(6)
SCORE_VIEWS = np.arange(6)
DINO_ANGLES = np.deg2rad([[0., 30.], [90., 30.], [180., 30.], [270., 30.], [0., 90.], [0., -90.]])
LAYOUT = dict(rows=3, columns=2, viewLabels=['FRONT', 'RIGHT', 'BACK', 'LEFT', 'TOP', 'BOTTOM'])


def reference_cameras(saved_frame, angles):
    """Frame and azimuth/elevation float64 [6,2] radians -> cameras float32 [6,4,4]/[6,3,3], normalized-to-world float64 [4,4]."""
    obb, scale = saved_frame['obb'], saved_frame['normalizationScale']
    P = np.eye(4)
    P[:3, :3] = np.asarray(obb['axes']).T / scale
    P[:3, 3] = obb['center']
    world_views, K = camera_matrices(obb, scale, angles, VIEW_SIZE)
    views = np.tile(np.eye(4), (len(angles), 1, 1))
    views[:, :3, :3] = world_views[:, :3, :3] @ (P[:3, :3] * scale)
    views[:, :3, 3] = (world_views[:, :3, :3] @ P[:3, 3] + world_views[:, :3, 3]) * scale
    assert np.allclose(views[:, :3, :3] @ views[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=1e-6)
    return views.astype(np.float32), K, P
