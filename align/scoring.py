"""The 8916 score: 40% partial-grid forward, 50% six-view IoU, 10% raw reverse."""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import torch
import trimesh

POLICY = dict(weights=dict(partialForward=.4, maskIoU=.5, rawReverse=.1),
              tolerance=.02, cellWidthFraction=.05, nearestFraction=.2, discardWorstFraction=.2,
              gaussianFPS=4096, meshSurfaceCandidates=65536, meshFPS=4096, seed=42,
              alphaThreshold=.5)


def partial_forward(points, distances, diagonal):
    """World points float64 [4096,3], normalized distances float64 [4096], diagonal scalar -> score and occupied/retained cell counts."""
    coordinates = np.floor((points - points.min(0)) / (.05 * diagonal)).astype(np.int64)
    _, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    errors = []
    for index in range(int(inverse.max()) + 1):
        local = np.sort(distances[inverse == index])
        nearest = int(np.ceil(.2 * len(local)))
        errors.append(float(np.median(local[:nearest])))
    retained = int(np.ceil(.8 * len(errors)))
    return dict(score=float(100 * np.mean(np.sort(errors)[:retained] <= .02)),
                occupiedCells=len(errors), retainedCells=retained)


def mask_iou(gaussian, mesh):
    """Gaussian/mesh alpha float32 [6,512,512] -> mean and per-view IoU on [0,100]; double-empty views excluded."""
    assert gaussian.shape == mesh.shape == (6, 512, 512)
    g, m = gaussian > .5, mesh > .5
    intersection = (g & m).sum((1, 2))
    union = (g | m).sum((1, 2))
    valid = union > 0
    assert valid.any(), 'No nonempty evaluation view'
    values = [float(100 * a / b) if b else None for a, b in zip(intersection, union)]
    return dict(score=float(np.mean([v for v in values if v is not None])), perView=values,
                validViews=int(valid.sum()), intersectionPixels=intersection.tolist(), unionPixels=union.tolist())


class AlignmentScorer:
    def __init__(self, points, device):
        """World Gaussian volume FPS float64 [4096,3], indexed CUDA device -> fixed scoring reference."""
        assert points.shape == (4096, 3) and np.isfinite(points).all()
        assert device.type == 'cuda' and device.index is not None
        self.points = points.astype(np.float64)
        self.device = device
        self.diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
        assert self.diagonal > 0
        self.origin = (points.min(0) + points.max(0)) / 2
        self.normalized = (self.points - self.origin) / self.diagonal
        self.tree = cKDTree(self.normalized)
        self.warmed = False

    def evaluate(self, mesh, world, gaussian_alpha, mesh_alpha):
        """Condition-space mesh, external world float64 [4,4], alphas float32 [6,512,512] -> weighted score and all three components."""
        vertices = np.asarray(mesh.vertices) @ world[:3, :3].T + world[:3, 3]
        vertices = (vertices - self.origin) / self.diagonal
        faces = np.asarray(mesh.faces)
        scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                            o3d.core.Tensor(faces.astype(np.uint32)))
        distances = scene.compute_distance(o3d.core.Tensor(self.normalized.astype(np.float32)), nthreads=1).numpy()
        forward = partial_forward(self.points, distances, self.diagonal)
        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        candidates = trimesh.sample.sample_surface(surface, 65536, seed=42)[0].astype(np.float32)
        with torch.cuda.device(self.device):
            from pytorch3d.ops import sample_farthest_points

            points = torch.as_tensor(candidates, device=self.device)[None]
            if not self.warmed:
                torch.cuda.empty_cache()
                sample_farthest_points(points, K=4096, random_start_point=False)
                torch.cuda.current_stream(self.device).synchronize()
                torch.cuda.empty_cache()
                self.warmed = True
            samples, _ = sample_farthest_points(points, K=4096, random_start_point=False)
            samples = samples[0].cpu().numpy()
        reverse = float(100 * np.mean(self.tree.query(samples, workers=1)[0] <= .02))
        masks = mask_iou(gaussian_alpha, mesh_alpha)
        score = .4 * forward['score'] + .5 * masks['score'] + .1 * reverse
        assert np.isfinite(score)
        return dict(score=float(score), partialForward=forward, maskIoU=masks, rawReverse=reverse,
                    referenceDiagonal=self.diagonal, toleranceWorld=.02 * self.diagonal)
