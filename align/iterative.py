"""Bounded similarity and fit-view contours, with correspondence refresh after each round."""
import time

import cv2
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation
import torch

from io_utils import write_json
from .matching import match_view
from .reference import FIT
from .rotation_math import LIMIT, matrix, project, solve
from .contour_math import unpack, alpha_render


def correspondences(matcher, reference, features, rgb, ga, mesh_rgb, ma, depth, views, K, parameters, view_weights):
    """RGB uint8 [V,512,512,3], alpha/depth float32 [V,512,512], parameters float64 [7] -> bbox XYZ [N,3], pixels [N,2], ids/weights [N], records/counts."""
    points, targets, ids, weights, records, counts = [], [], [], [], [], []
    inverse_delta = np.linalg.inv(matrix(parameters))
    combined = [torch.cat([a, b]) for a, b in zip(reference, features)]
    for i, v in enumerate(FIT):
        matched = match_view(matcher, combined, i, v,
            rgb[i], mesh_rgb[i], ga[i], ma[i], depth[i])
        counts.append(matched.info)
        a, b = matched.mesh_pixels, matched.gaussian_pixels
        if not len(a):
            continue
        z = cv2.remap(depth[i], a[:, 0][None], a[:, 1][None], cv2.INTER_NEAREST)[0]
        assert np.all(z > 0)
        rays = np.column_stack([(a+.5-K[v, :2, 2])/[K[v, 0, 0], K[v, 1, 1]], np.ones(len(a))])
        xyz = (rays*z[:, None]-views[v, :3, 3]) @ views[v, :3, :3]
        points.extend(xyz @ inverse_delta[:3, :3].T+inverse_delta[:3, 3])
        targets.extend(b+.5)
        ids.extend([i]*len(a))
        weights.extend(matched.weights)
        records.extend(matched.records)
    points, targets = np.asarray(points, np.float64).reshape(-1, 3), np.asarray(targets, np.float64).reshape(-1, 2)
    ids, weights = np.asarray(ids, np.int64), np.asarray(weights, np.float64)
    if len(weights):
        weights /= np.bincount(ids, minlength=len(FIT))[ids]
        weights *= view_weights[ids]
        weights /= np.median(weights)
    return points, targets, ids, weights, records, counts


def fit_iterative(case, matcher, folder, rgb, ga, mesh_rgb, mesh_alpha, mesh_depth, views, K, renderer, C, *, contour, original_alpha, source, view_weights):
    """RGB uint8 [V,512,512,3], alpha/depth float32 [V,512,512], cameras float32 [V,4,4]/[V,3,3], bbox matrix float64 [4,4] -> correction [4,4], errors [N] and fit records."""
    device = matcher.device
    mask = contour > .5
    active_views = mask.sum((1, 2)) >= 64
    assert active_views.any(), f'{case}: no observed contour reference'
    assert view_weights.shape == (len(FIT),) and (view_weights > 0).all()
    parameters = np.array([1., 0., 0., 0., 0., 0., 0.])
    reference_started = time.perf_counter()
    reference = matcher.pyramid(rgb)
    reference_seconds = time.perf_counter()-reference_started
    signed = np.stack([cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        -cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for m in mask])
    with torch.cuda.device(device):
        sdf = torch.as_tensor(np.clip(signed/32, -1, 1), device=device)
        target = torch.as_tensor(contour, dtype=torch.float32, device=device)
        full_reference = torch.tensor([s == 'FAL' for s in source], device=device)
        view_weight = torch.as_tensor(active_views*view_weights, dtype=torch.float32, device=device)
        allowed = torch.as_tensor(np.stack([cv2.dilate((m > .15).astype(np.uint8), np.ones((15, 15), np.uint8)) for m in original_alpha]), dtype=torch.float32, device=device)
        bbox = torch.as_tensor(C, dtype=torch.float32, device=device)
        vertices = renderer.vertices @ bbox[:3, :3].T+bbox[:3, 3]
        def encode(p):
            """parameters float64 [7] -> trainable raw float32 [7] on owning device."""
            angle = np.linalg.norm(p[4:])
            rotation = p[4:]/max(angle, 1e-8)*np.arctanh(min(angle/LIMIT, .9999))
            return torch.tensor(np.r_[np.arctanh(np.clip((p[0]-1)/.2, -.9999, .9999)),
                np.arctanh(np.clip(p[1:4]/.15, -.9999, .9999)), rotation], dtype=torch.float32, device=device, requires_grad=True)
        def shape_and_prior(raw):
            """raw trainable float32 [7] -> category-independent shape and pose-prior scalar loss."""
            alpha = alpha_render(renderer, vertices, raw)
            area = alpha.sum((1, 2)).clamp_min(1)
            intersection = (alpha*target).sum((1, 2))
            full = .2*(1-intersection/(area+target.sum((1, 2))-intersection).clamp_min(1))
            full = full+(alpha*sdf).sum((1, 2))/target.sum((1, 2)).clamp_min(1)
            coverage = 1-intersection/target.sum((1, 2)).clamp_min(1)
            outside = (alpha*(1-allowed)).sum((1, 2))/area
            per_view = torch.where(full_reference, full, coverage+.25*outside)
            shape = (per_view*view_weight).sum()/view_weight.sum()
            scale, offset, _, rotation = unpack(raw)
            prior = .04*((scale-1)/.2)**2+.02*((offset/.15)**2).sum()
            prior = prior+.003*((rotation[[0, 2]]/np.deg2rad(5))**2).sum()+.003*((rotation/LIMIT)**2).sum()
            return shape+prior
        initialization_started = time.perf_counter()
        starts = [parameters.copy()]
        for axis in range(3):
            for degrees in (-29.99, -15., 15., 29.99):
                p = parameters.copy()
                p[4+axis] = np.deg2rad(degrees)
                starts.append(p)
        coarse = []
        for n, start in enumerate(starts):
            raw = encode(start)
            optimizer = torch.optim.Adam([raw], lr=.018)
            value = shape_and_prior(raw).item()
            best, refined = value, start.copy()
            for _ in range(12):
                optimizer.zero_grad(set_to_none=True)
                loss = shape_and_prior(raw)
                value = loss.item()
                if value < best:
                    best = value
                    scale, offset, _, rotation = unpack(raw)
                    refined = np.r_[scale.item(), offset.cpu().tolist(), rotation.cpu().tolist()]
                loss.backward()
                optimizer.step()
            coarse.append((best, n, refined))
        scored = sorted(coarse, key=lambda item:item[0])
        best_initial_loss = scored[0][0]
        parameters = scored[0][2].copy()
        initialization_records = []
        for initial_loss, index, start in scored[:3]:
            raw = encode(start)
            optimizer = torch.optim.Adam([raw], lr=.018)
            best = initial_loss
            for _ in range(40):
                optimizer.zero_grad(set_to_none=True)
                loss = shape_and_prior(raw)
                value = loss.item()
                if value < best_initial_loss:
                    best_initial_loss = value
                    scale, offset, _, rotation = unpack(raw)
                    parameters = np.r_[scale.item(), offset.cpu().tolist(), rotation.cpu().tolist()]
                best = min(best, value)
                loss.backward()
                optimizer.step()
            initialization_records.append(dict(startIndex=index, start=start.tolist(), initialLoss=initial_loss, finalLoss=best))
        initialization_seconds = time.perf_counter()-initialization_started
        write_json(folder/'orientation-search.json', dict(coarseStarts=13, coarseStepsPerStart=12, refinedStarts=3, stepsPerStart=40,
            usesObjectCategory=False, usesPartLabels=False, candidates=initialization_records,
            chosen=parameters.tolist(), seconds=initialization_seconds))
        rounds = []
        best_loss = None
        for iteration in range(1, 4):
            started = time.perf_counter()
            mesh_rgb, mesh_alpha, mesh_depth = renderer.render(matrix(parameters) @ C, FIT)
            features = matcher.pyramid(mesh_rgb)
            points, pixels, ids, weights, records, counts = correspondences(matcher, reference, features,
                rgb, ga, mesh_rgb, mesh_alpha, mesh_depth, views, K, parameters, view_weights)
            matching_seconds = time.perf_counter()-started
            started = time.perf_counter()
            status, diagnostics = 'insufficient_matches', {}
            candidate = parameters.copy()
            if len(points) >= 12 and len(np.unique(ids)) >= 2:
                candidate, diagnostics = solve(points, pixels, ids, weights, views[FIT], K[FIT], parameters, True)
                status = diagnostics['status']
            solve_seconds = time.perf_counter()-started
            errors = np.linalg.norm(project(points, candidate, views[FIT], K[FIT], ids)-pixels, axis=1)
            consensus = errors < 8
            xyz = torch.as_tensor(points[consensus], dtype=torch.float32, device=device)
            uv = torch.as_tensor(pixels[consensus], dtype=torch.float32, device=device)
            vi = torch.as_tensor(ids[consensus], device=device)
            weight = torch.as_tensor(weights[consensus], dtype=torch.float32, device=device)
            use_features = len(xyz) >= 6 and len(np.unique(ids[consensus])) >= 2
            camera_ids = torch.as_tensor(FIT, device=device)[vi]
            camera_R, camera_t = renderer.views[camera_ids, :3, :3], renderer.views[camera_ids, :3, 3]
            camera_k = renderer.intrinsics[camera_ids]
            def objective(raw):
                """raw trainable float32 [7] -> scalar contour + reprojection + bounded-pose prior loss."""
                scale, offset, R, rotation = unpack(raw)
                feature = torch.zeros((), device=device)
                if use_features:
                    transformed = scale*(xyz @ R.T)+offset
                    camera = torch.einsum('nij,nj->ni', camera_R, transformed)+camera_t
                    projected = camera[:, :2]/camera[:, 2:]*camera_k[:, [0, 1], [0, 1]]+camera_k[:, :2, 2]
                    error2 = ((projected-uv)**2).sum(1)
                    feature = (torch.log1p(error2/16)*weight).sum()/weight.sum()
                return shape_and_prior(raw)+.05*float(consensus.mean() if len(consensus) else 0)*feature
            started = time.perf_counter()
            # Compare both starts under this round's SAME objective and correspondence set.
            starts = [parameters, candidate]
            start_values = [objective(encode(p)).item() for p in starts]
            raw = encode(starts[int(np.argmin(start_values))])
            best_loss = min(start_values)
            best_parameters = starts[int(np.argmin(start_values))].copy()
            initial_loss = best_loss
            optimizer = torch.optim.Adam([raw], lr=.012)
            previous = parameters.copy()
            for _ in range(100):
                optimizer.zero_grad(set_to_none=True)
                loss = objective(raw)
                value = loss.item()
                assert np.isfinite(value), f'{case}: nonfinite joint objective'
                if value < best_loss:
                    scale, offset, _, rotation = unpack(raw)
                    best_loss = value
                    best_parameters = np.r_[scale.item(), offset.cpu().tolist(), rotation.cpu().tolist()]
                loss.backward()
                optimizer.step()
            torch.cuda.current_stream(device).synchronize()
            contour_seconds = time.perf_counter()-started
            parameters = best_parameters.copy()
            step = float(np.linalg.norm(parameters-previous))
            row = dict(iteration=iteration, correspondenceStatus=status, matches=len(points), jointFeatureInliers=int(consensus.sum()),
                featureConstraintUsed=use_features, matchingSeconds=matching_seconds, solveSeconds=solve_seconds, contourSeconds=contour_seconds,
                loss=best_loss, lossImprovement=initial_loss-best_loss, parameters=parameters.tolist(), parameterStep=step, candidateDiagnostics=diagnostics)
            rounds.append(row)
            write_json(folder/'iterations.json', dict(case=case, maximumRotationDegrees=30, maximumRounds=3,
                contourStepsPerRound=100, gaussianFeatureSeconds=reference_seconds, rounds=rounds,
                selection='joint reference contour + robust feature reprojection + pose prior; previous pose re-evaluated each round', reference=source))
            print('ITERATION', case, row, flush=True)
            if step < 1e-5:
                break
        assert np.linalg.norm(parameters[4:]) <= LIMIT+1e-6
    # Refresh once at the final pose so displayed matches and errors describe the exported model.
    started = time.perf_counter()
    mesh_rgb, mesh_alpha, mesh_depth = renderer.render(matrix(parameters) @ C, FIT)
    features = matcher.pyramid(mesh_rgb)
    points, pixels, ids, weights, records, counts = correspondences(matcher, reference, features,
        rgb, ga, mesh_rgb, mesh_alpha, mesh_depth, views, K, parameters, view_weights)
    refresh_seconds = time.perf_counter()-started
    errors = np.linalg.norm(project(points, parameters, views[FIT], K[FIT], ids)-pixels, axis=1)
    initial_errors = np.linalg.norm(project(points, np.array([1., 0., 0., 0., 0., 0., 0.]), views[FIT], K[FIT], ids)-pixels, axis=1)
    good = errors < 6
    condition_number = 0.
    if good.sum() >= 4:
        jacobian = approx_derivative(lambda p: project(points[good], p, views[FIT], K[FIT], ids[good]).ravel(), parameters)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition_number = float(singular[0]/max(singular[-1], 1e-12))
    fit = dict(status='completed', reason='', matches=len(points), inliers=int(good.sum()),
        activeContourViews=FIT[active_views].tolist(), disabledContourViews=FIT[~active_views].tolist(),
        referenceSources=source, viewWeights=view_weights.tolist(),
        matchesPerView=np.bincount(ids, minlength=len(FIT)).tolist(),
        inliersPerView=np.bincount(ids[good], minlength=len(FIT)).tolist(), conditionNumber=condition_number,
        medianReprojectionPixels=float(np.median(errors)) if len(errors) else None,
        inlierMedianPixels=float(np.median(errors[good])) if good.any() else None,
        rotationDegrees=float(np.rad2deg(np.linalg.norm(parameters[4:]))),
        rotationEulerDegrees=Rotation.from_rotvec(parameters[4:]).as_euler('xyz', degrees=True).tolist(),
        rotationMatrix=Rotation.from_rotvec(parameters[4:]).as_matrix().tolist(),
        verticalTiltDegrees=float(np.rad2deg(np.arccos(np.clip(Rotation.from_rotvec(parameters[4:]).as_matrix()[1, 1], -1, 1)))),
        rounds=len(rounds), contourLoss=best_loss, gaussianFeatureSeconds=reference_seconds,
        finalCorrespondenceRefreshSeconds=refresh_seconds,
        orientationInitializationSeconds=initialization_seconds,
        coreSeconds=initialization_seconds+reference_seconds+refresh_seconds+sum(r['matchingSeconds']+r['solveSeconds']+r['contourSeconds'] for r in rounds))
    np.savez_compressed(folder/'correspondences.npz', points=points, pixels=pixels, ids=ids, weights=weights, views=views[FIT], K=K[FIT])
    return matrix(parameters), errors, initial_errors, fit, records, counts
