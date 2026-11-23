from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

SH_C0 = 0.28209479177387814
RESOLUTION = 16


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Binary Gaussian PLY -> float64 xyz/scale [N,3], wxyz [N,4], opacity [N]; structured float32 records [N]."""
    body = path.read_bytes()
    marker = body.find(b"end_header")
    assert marker >= 0, path
    offset = body.find(b"\n", marker) + 1
    header = body[:offset].decode("ascii")
    assert "format binary_little_endian 1.0" in header
    count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex ")))
    names = [line.split()[2] for line in header.splitlines() if line.startswith("property float ")]
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
                "f_dc_0", "f_dc_1", "f_dc_2"}
    assert required <= set(names), sorted(required - set(names))
    dtype = np.dtype([(name, "<f4") for name in names])
    records = np.frombuffer(body, dtype=dtype, count=count, offset=offset).copy()
    xyz = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(np.float64)
    logits = records["opacity"].astype(np.float64)
    opacity = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    scale = np.exp(np.column_stack([records[f"scale_{i}"] for i in range(3)]).astype(np.float64))
    rotation = np.column_stack([records[f"rot_{i}"] for i in range(4)]).astype(np.float64)
    color = np.clip(.5 + SH_C0 * np.column_stack([records[f"f_dc_{i}"] for i in range(3)]), 0., 1.)
    return {"xyz": xyz, "opacity": opacity, "scale": scale, "rotation": rotation, "color": color, "records": records}


def write_ply(source: Path, records: np.ndarray, output: Path) -> None:
    """records: structured float32 Gaussian array [N], preserving the source property order."""
    body = source.read_bytes()
    marker = body.find(b"end_header")
    offset = body.find(b"\n", marker) + 1
    header = body[:offset].decode("ascii")
    old = next(line for line in header.splitlines() if line.startswith("element vertex "))
    header = header.replace(old, f"element vertex {len(records)}", 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header.encode("ascii") + records.tobytes())


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """q: float64 [N,4] wxyz; returns float64 [N,3,3]."""
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    assert np.isfinite(norm).all() and np.all(norm > 1e-12)
    q = q / norm
    w, x, y, z = q.T
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def covariance_stats(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """data: float64 scale [N,3], rotation [N,4]; returns covariance [N,3,3], radius/ratio [N]."""
    rotation = quaternion_matrix(data["rotation"])
    covariance = np.einsum("nij,nj,nkj->nik", rotation, data["scale"] ** 2, rotation)
    eigenvalues = np.linalg.eigvalsh(covariance)
    radius = 3.0 * np.sqrt(np.maximum(eigenvalues[:, -1], 1e-18))
    anisotropy = eigenvalues[:, -1] / np.maximum(eigenvalues[:, 0], 1e-18)
    return covariance, radius, anisotropy


def covariance_prefilter(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """data: aligned Gaussian arrays [N,...]; returns selected arrays [M,...] and counts."""
    xyz, opacity, scale = data["xyz"], data["opacity"], data["scale"]
    finite = np.isfinite(xyz).all(1) & np.isfinite(scale).all(1) & np.isfinite(data["rotation"]).all(1) & np.isfinite(opacity)
    assert finite.any()
    alpha_cut = .01
    alpha_keep = finite & (opacity >= alpha_cut)
    radius = np.full(len(xyz), np.inf)
    anisotropy = np.full(len(xyz), np.inf)
    _, radius[finite], anisotropy[finite] = covariance_stats({key: value[finite] for key, value in data.items()})
    eligible = alpha_keep & np.isfinite(radius) & np.isfinite(anisotropy)
    bounds = np.quantile(xyz[eligible], [.005, .995], axis=0)
    object_diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    r_limit = 4 * object_diagonal
    a_limit = float(np.quantile(np.log(np.maximum(anisotropy[eligible], 1.)), .9975))
    # Thin, broad Gaussians are valid surface representations, including tabletops.
    # Anisotropy is diagnostic; only covariance extending far beyond the object is rejected.
    covariance_keep = eligible & (radius <= r_limit)
    selected_xyz = xyz[covariance_keep]
    low = np.quantile(selected_xyz, .0005, axis=0)
    high = np.quantile(selected_xyz, .9995, axis=0)
    margin = max(float(np.linalg.norm(high - low)) * .25, 1e-8)
    spatial_keep = np.all((xyz >= low - margin) & (xyz <= high + margin), axis=1)
    keep = covariance_keep & spatial_keep
    assert keep.any(), "No Gaussian survived filtering"
    filtered = {key: value[keep] for key, value in data.items()}
    return filtered, {"sourceCount": int(len(xyz)), "retainedCount": int(keep.sum()), "alphaCut": alpha_cut,
                      "covarianceRadiusLimit": r_limit, "covarianceAnisotropyLogQuantile9975": a_limit, "anisotropyUsedForRemoval": False, "objectDiagonal": object_diagonal,
                      "covarianceRemoved": int((~covariance_keep).sum()), "spatialRemoved": int((covariance_keep & ~spatial_keep).sum())}

def convex_hull(points: np.ndarray) -> np.ndarray:
    """XZ float64 [N,2] -> convex hull vertices float64 [M,2]."""
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) <= 2:
        return np.asarray(unique)

    def cross(origin: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def compute_semantic_obb(xyz: np.ndarray) -> dict[str, Any]:
    """World XYZ float64 [N,3] -> robust OBB with source -Y up, +X front, +Z right."""
    # Source convention is fixed: -Y up, +X front, +Z right.
    # Candidate yaw only minimizes the horizontal footprint; sign and axis order
    # are then anchored to +X so front/back cannot randomly flip.
    sample = xyz
    if len(sample) > 100_000:
        sample = sample[np.linspace(0, len(sample) - 1, 100_000, dtype=int)]
    hull = convex_hull(sample[:, [0, 2]])
    angles = [0.0] if len(hull) < 2 else [
        math.atan2(hull[(index + 1) % len(hull), 1] - point[1], hull[(index + 1) % len(hull), 0] - point[0])
        for index, point in enumerate(hull)
    ]
    best = None
    for angle in angles:
        c, s = math.cos(angle), math.sin(angle)
        candidate = np.asarray([[c, 0, s], [s, 0, -c]])
        projected = sample @ candidate.T
        size = np.quantile(projected, 0.995, axis=0) - np.quantile(projected, 0.005, axis=0)
        area = float(size.prod())
        if best is None or area < best[0]:
            best = (area, candidate)
    assert best is not None
    horizontal = best[1]
    front = horizontal[0] if abs(horizontal[0, 0]) >= abs(horizontal[1, 0]) else horizontal[1]
    if front[0] < 0:
        front = -front
    up = np.asarray([0.0, -1.0, 0.0])
    right = np.asarray([-front[2], 0.0, front[0]])
    axes = np.asarray([right, up, front])
    projected = xyz @ axes.T
    minimum = np.quantile(projected, 0.0025, axis=0)
    maximum = np.quantile(projected, 0.9975, axis=0)
    local_center = (minimum + maximum) / 2
    center = axes.T @ local_center
    size = maximum - minimum
    return {
        "algorithm": "robust-minimum-area-xz; source -Y-up +X-front +Z-right",
        "center": center.tolist(),
        "axes": axes.tolist(),
        "size": size.tolist(),
        "yawRadians": float(math.atan2(front[2], front[0])),
        "sourceConvention": {"up": "-Y", "front": "+X", "right": "+Z"},
        "conditionConvention": {"x": "right", "y": "up", "z": "front"},
    }


def normalize_gaussians(data: dict[str, np.ndarray], obb: dict[str, Any], extent: float) -> dict[str, np.ndarray]:
    """Gaussian float64 arrays [N,...] -> OBB-normalized xyz/scale float64 [N,3]."""
    axes = np.asarray(obb["axes"])
    center = np.asarray(obb["center"])
    factor = extent / max(obb["size"])
    return {
        **data,
        "xyz": (data["xyz"] - center) @ axes.T * factor,
        "scale": data["scale"] * factor,
    }


def normalized_covariances(
    data: dict[str, np.ndarray], obb: dict[str, Any], extent: float
) -> np.ndarray:
    """Gaussian float64 arrays [N,...] -> OBB-normalized covariance float64 [N,3,3]."""
    rotation = quaternion_matrix(data["rotation"])
    axes = np.asarray(obb["axes"], dtype=np.float64)
    factor = extent / max(obb["size"])
    condition_rotation = np.einsum("ij,njk->nik", axes, rotation)
    normalized_scale = data["scale"] * factor
    return np.einsum(
        "nij,nj,nkj->nik",
        condition_rotation,
        normalized_scale**2,
        condition_rotation,
    )


def rasterize_voxel(data: dict[str, np.ndarray], covariances: np.ndarray) -> np.ndarray:
    """OBB-normalized Gaussian arrays [N,...], covariance float64 [N,3,3] -> occupancy bool [16,16,16]."""
    source_resolution = 64
    opacity_threshold = .5
    grid = np.zeros((source_resolution,) * 3, dtype=bool)
    opacity_keep = data["opacity"] >= opacity_threshold
    eligible_indices = np.flatnonzero(opacity_keep)
    centers_normalized = data["xyz"][eligible_indices]
    # Keep one empty 16^3 cell around the condition. The robust OBB deliberately
    # tolerates a few spatial tails, which must not leak onto the grid boundary.
    condition_limit = 0.5 - 1 / RESOLUTION
    spatial_keep = np.all(
        (centers_normalized >= -condition_limit) & (centers_normalized < condition_limit),
        axis=1,
    )
    centers = centers_normalized[spatial_keep] * source_resolution + source_resolution / 2
    voxels = np.floor(centers).astype(int)
    valid = np.all((voxels >= 0) & (voxels < source_resolution), axis=1)
    grid[tuple(voxels[valid].T)] = True

    indices = eligible_indices[spatial_keep]
    border = source_resolution // RESOLUTION
    maximum_index = source_resolution - border - 1
    for index in indices:
        opacity = float(data["opacity"][index])
        mahalanobis_limit = 2 * math.log(opacity / opacity_threshold)
        if mahalanobis_limit <= 1e-12:
            continue
        covariance = covariances[index]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        inverse = (eigenvectors * (1 / eigenvalues)) @ eigenvectors.T
        center = data["xyz"][index]
        radius = np.sqrt(np.maximum(np.diag(covariance), 0) * mahalanobis_limit)
        minimum = np.ceil((center - radius + 0.5) * source_resolution - 0.5).astype(int)
        maximum = np.floor((center + radius + 0.5) * source_resolution - 0.5).astype(int)
        minimum = np.maximum(minimum, border)
        maximum = np.minimum(maximum, maximum_index)
        if np.any(minimum > maximum):
            continue
        axes = [np.arange(minimum[axis], maximum[axis] + 1) for axis in range(3)]
        coordinates = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        points = (coordinates + 0.5) / source_resolution - 0.5
        delta = points - center
        distance = np.einsum("...i,ij,...j->...", delta, inverse, delta)
        occupied = distance <= mahalanobis_limit
        if occupied.any():
            region = grid[
                minimum[0] : maximum[0] + 1,
                minimum[1] : maximum[1] + 1,
                minimum[2] : maximum[2] + 1,
            ]
            region |= occupied

    factor = source_resolution // RESOLUTION
    fine_support = grid.reshape(
        RESOLUTION,
        factor,
        RESOLUTION,
        factor,
        RESOLUTION,
        factor,
    ).sum(axis=(1, 3, 5))
    support_threshold = 1 + int(np.count_nonzero(fine_support >= 2) >= 16)
    result = fine_support >= support_threshold

    # A Gaussian reconstruction describes visible surfaces, including thin open
    # structures. Fill only truly enclosed 3D cavities; projection-based filling
    # incorrectly turns tables, chairs, and open laptops into solid cuboids.
    labels, count = ndimage.label(result, structure=ndimage.generate_binary_structure(3, 1))
    if count > 1:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        result = labels == sizes.argmax()
    result = ndimage.binary_fill_holes(result)
    return result
