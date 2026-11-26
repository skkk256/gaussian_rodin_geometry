import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
from .matching import fit_reprojection
LIMIT = np.deg2rad(30.)

def matrix(parameters):
    """parameters float64 [7] -> bbox-relative similarity float64 [4,4]."""
    result = np.eye(4)
    result[:3, :3] = parameters[0]*Rotation.from_rotvec(parameters[4:]).as_matrix()
    result[:3, 3] = parameters[1:4]
    return result

def project(points, parameters, views, K, ids):
    """points float64 [N,3], parameters [7], camera arrays and ids [N] -> pixels float64 [N,2]."""
    transform = matrix(parameters)
    xyz = points @ transform[:3, :3].T+transform[:3, 3]
    camera = np.einsum('nij,nj->ni', views[ids, :3, :3], xyz)+views[ids, :3, 3]
    return camera[:, :2]/camera[:, 2:]*K[ids][:, [0, 1], [0, 1]]+K[ids, :2, 2]

def solve(points, pixels, ids, weights, views, K, previous, rotation):
    """3D points float64 [N,3], target pixels [N,2], ids/weights [N], previous [7] -> bounded parameters [7], diagnostics."""
    if not rotation:
        fit = fit_reprojection(points, pixels, ids, weights, views, K)
        assert fit.status == 'completed', fit.reason
        return np.r_[fit.parameters, np.zeros(3)], fit.statistics
    assert len(points) >= 12 and len(np.unique(ids)) >= 2
    counts = np.bincount(ids, minlength=len(views))
    def errors(p):
        return np.linalg.norm(project(points, p, views, K, ids)-pixels, axis=1)
    def score(p):
        e = errors(p)
        return np.sum(np.maximum(0., 1-(e/6)**2)*weights/np.sqrt(counts[ids]))
    def residual(p):
        return ((project(points, p, views, K, ids)-pixels)*np.sqrt(weights[:, None])).ravel()
    # Affine multi-view hypotheses provide fast rotation-aware RANSAC initialization.
    camera_R, camera_t = views[ids, :3, :3], views[ids, :3, 3]
    uv = (pixels-K[ids, :2, 2])/K[ids][:, [0, 1], [0, 1]]
    E = camera_R[:, :2]-uv[:, :, None]*camera_R[:, 2:3]
    A = np.concatenate([np.einsum('nki,nj->nkij', E, points).reshape(-1, 2, 9), E], axis=2)
    rhs = uv*camera_t[:, 2:]-camera_t[:, :2]
    candidates = [previous.copy(), np.array([1., 0., 0., 0., 0., 0., 0.])]
    rng = np.random.default_rng(20261125)
    for _ in range(500):
        sample = rng.choice(len(points), 8, replace=False, p=weights/weights.sum())
        if len(np.unique(ids[sample])) < 2:
            continue
        affine, _, rank, _ = np.linalg.lstsq(A[sample].reshape(-1, 12), rhs[sample].ravel(), rcond=None)
        if rank < 12:
            continue
        U, singular, V = np.linalg.svd(affine[:9].reshape(3, 3))
        D = np.diag([1., 1., np.linalg.det(U @ V)])
        R = U @ D @ V
        scale = float(np.mean(singular*np.diag(D)))
        rotvec = Rotation.from_matrix(R).as_rotvec()
        if not (.8 < scale < 1.2 and np.linalg.norm(rotvec) <= LIMIT):
            continue
        rotated = points @ (scale*R).T
        offset = np.linalg.lstsq(E[sample].reshape(-1, 3),
            (rhs-np.einsum('nki,ni->nk', E, rotated))[sample].ravel(), rcond=None)[0]
        if np.max(np.abs(offset)) < .15:
            candidates.append(np.r_[scale, offset, rotvec])
    # Include small rotations around every axis; all starts optimize all seven parameters.
    for axis in range(3):
        for degrees in (-20., -10., 10., 20.):
            p = previous.copy()
            p[4+axis] += np.deg2rad(degrees)
            p[4:] *= min(1., LIMIT/max(np.linalg.norm(p[4:]), 1e-12))
            candidates.append(p)
    candidates.sort(key=score, reverse=True)
    lower, upper = np.r_[.8, [-.15]*3, [-LIMIT]*3], np.r_[1.2, [.15]*3, [LIMIT]*3]
    solutions = []
    for start in candidates[:8]:
        fit = least_squares(residual, start, bounds=(lower, upper), loss='cauchy', f_scale=3., max_nfev=70)
        p = fit.x
        if np.linalg.norm(p[4:]) > LIMIT:
            p[4:] *= LIMIT/np.linalg.norm(p[4:])
            fit2 = minimize(lambda x: float(np.sum(9*np.log1p((residual(x)/3)**2))), p,
                method='SLSQP', bounds=list(zip(lower, upper)),
                constraints=[dict(type='ineq', fun=lambda x: LIMIT**2-np.dot(x[4:], x[4:]))],
                options=dict(maxiter=100, ftol=1e-8))
            if not fit2.success:
                continue
            p = fit2.x
        solutions.append(p)
    chosen = max(solutions+[previous], key=score).copy()
    for _ in range(3):
        good = errors(chosen) < 6
        if good.sum() < 6 or len(np.unique(ids[good])) < 2:
            return previous.copy(), dict(status='insufficient_consensus', matches=len(points), inliers=int(good.sum()))
        def inlier_residual(p):
            return ((project(points[good], p, views, K, ids[good])-pixels[good])*np.sqrt(weights[good, None])).ravel()
        fit = least_squares(inlier_residual, chosen, bounds=(lower, upper), loss='cauchy', f_scale=1.5, max_nfev=70)
        if np.linalg.norm(fit.x[4:]) > LIMIT or score(fit.x) < score(chosen):
            break
        chosen = fit.x
    assert np.linalg.norm(chosen[4:]) <= LIMIT+1e-7
    e = errors(chosen)
    good = e < 6
    return chosen, dict(status='completed', matches=len(points), inliers=int(good.sum()), inlierMedianPixels=float(np.median(e[good])),
        medianReprojectionPixels=float(np.median(e)), inliersPerView=np.bincount(ids[good], minlength=len(views)).tolist(),
        admissibleCandidates=len(candidates), rotationDegrees=float(np.rad2deg(np.linalg.norm(chosen[4:]))),
        rotationEulerDegrees=Rotation.from_rotvec(chosen[4:]).as_euler('xyz', degrees=True).tolist())
