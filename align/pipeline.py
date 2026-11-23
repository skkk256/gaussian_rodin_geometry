"""Complete inverse -> filtered-Gaussian FPS bbox -> eight-view DINOv3/SIFT."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import trimesh

from conditions import PointSampler, normalize_points
from gaussian import compute_semantic_obb, read_ply
from io_utils import sha256, write_json, write_rgb
from .geometry import register_inverse_fps, write_glb
from .matching import MATCHING_VERSION, fit_reprojection, match_view, projection
from .rendering import CHECK, FIT, MeshRenderer, fitted_cameras, gaussian_render


def input_hashes_for(result: Path, work: Path) -> dict:
    config = json.loads((work/'config.json').read_text())
    paths = dict(mesh=result/'rodin_textured.glb', inverse=result/'transform.json',
                 filtered=result/'gaussian_filter.ply', condition=work/'condition.npz',
                 canonical=work/'canonical.json', original=Path(config['source']))
    hashes = {key: sha256(path) for key, path in paths.items()}
    assert hashes['original'] == config['sourceSha256'], 'Original Gaussian changed'
    assert hashes['filtered'] == sha256(work/'gaussian_filter.ply'), 'Result/work filtered Gaussians differ'
    return hashes


def metric(alpha, reference):
    """alpha/reference float32 [4,512,512] -> held-out IoU values, mean and worst."""
    a, b = alpha > .5, reference > .5
    values = (a & b).sum((1, 2))/(a | b).sum((1, 2))
    return dict(viewIds=CHECK.tolist(), perViewIoU=values.tolist(), meanIoU=float(values.mean()), worstIoU=float(values.min()))


def fit_images(case, matcher, folder, rgb, ga, mesh_rgb, mesh_alpha, mesh_depth, views, K):
    """RGB uint8 [8,512,512,3], alpha/depth float32 [8,512,512], cameras float32 [18,4,4]/[18,3,3] -> delta float64 [4,4], errors float64 [N], fit and correspondence records."""
    xyz, pixels, ids, weights, records, counts = [], [], [], [], [], []
    for start in (0, 4):
        features = matcher.pyramid(np.concatenate([rgb[start:start+4], mesh_rgb[start:start+4]]))
        for ordinal in range(4):
            fit_id = start+ordinal
            index = FIT[fit_id]
            matches = match_view(matcher, features, ordinal, index,
                rgb[fit_id], mesh_rgb[fit_id], ga[fit_id], mesh_alpha[fit_id], mesh_depth[fit_id])
            counts.append(matches.info)
            print('MATCH', case, matches.info, flush=True)
            a, b = matches.mesh_pixels, matches.gaussian_pixels
            if not len(a):
                continue
            z = cv2.remap(mesh_depth[fit_id], a[:, 0][None], a[:, 1][None], cv2.INTER_NEAREST)[0]
            assert np.all(z > 0), f'{case} 视角 {index} 的匹配点没有有效网格深度'
            rays = np.column_stack([(a+.5-K[index, :2, 2])/[K[index, 0, 0], K[index, 1, 1]], np.ones(len(a))])
            xyz.extend((rays*z[:, None]-views[index, :3, 3]) @ views[index, :3, :3])
            pixels.extend(b+.5)
            weights.extend(matches.weights)
            ids.extend([fit_id]*len(a))
            records.extend(matches.records)
        del features
    xyz, pixels = np.asarray(xyz, np.float64).reshape(-1, 3), np.asarray(pixels, np.float64).reshape(-1, 2)
    ids, weights = np.asarray(ids, np.int64), np.asarray(weights, np.float64)
    if len(weights):
        weights /= np.bincount(ids, minlength=len(FIT))[ids]
        weights /= np.median(weights)
    np.savez_compressed(folder/'correspondences.npz', points=xyz, pixels=pixels, ids=ids, weights=weights, views=views[FIT], K=K[FIT])
    solution = fit_reprojection(xyz, pixels, ids, weights, views[FIT], K[FIT])
    initial_errors = np.linalg.norm(projection(xyz, np.array([1., 0., 0., 0.]), views[FIT], K[FIT], ids)-pixels, axis=1)
    delta = np.eye(4)
    if solution.status == 'completed':
        delta[:3, :3] *= solution.parameters[0]
        delta[:3, 3] = solution.parameters[1:]
    fit = dict(solution.statistics, status=solution.status, reason=solution.reason)
    return delta, solution.errors, initial_errors, fit, records, counts


def align_case(inputs: Path, work: Path, folder: Path, matcher, sampler: PointSampler,
               device: torch.device, *, per_view: bool = False, texture_slot: str = 'emissive') -> dict:
    assert device.type == 'cuda' and device.index is not None
    assert texture_slot in ('emissive', 'base-color')
    assert not folder.is_relative_to(inputs) and not folder.is_relative_to(work), 'Keep alignment output separate from generation inputs'
    started = time.monotonic()
    input_hashes = input_hashes_for(inputs, work)
    source = inputs/'rodin_textured.glb'
    config = json.loads((work/'config.json').read_text())
    case = config['sceneId']+'/'+config['objectId']
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'result.json').unlink(missing_ok=True)
    data = read_ply(inputs/'gaussian_filter.ply')
    transform = json.loads((inputs/'transform.json').read_text())
    assert transform['matrixOrder'] == 'column-major'
    inverse = np.asarray(transform['rodinToWorld']['matrix'], np.float64).reshape(4, 4, order='F')
    saved_frame = json.loads((work/'canonical.json').read_text())
    canonical_to_world = np.asarray(saved_frame['canonicalToWorld'], np.float64).reshape(4, 4, order='F')
    with np.load(work/'condition.npz') as saved:
        canonical_points = saved['canonical']
        condition = saved['points']
        normalization = saved['normalization']
    assert condition.shape == canonical_points.shape == (4096, 3)
    assert np.allclose(inverse, canonical_to_world @ np.linalg.inv(normalization), atol=1e-9, rtol=1e-9), 'transform.json must contain the complete inverse without bbox/DINO alignment'
    sampled = sampler.sample(data, canonical_to_world)
    sample_error = float(np.max(np.abs(sampled-canonical_points)))
    assert sample_error < 1e-6, f'{case}: filtered-Gaussian FPS differs from generation: {sample_error}'
    normalized, verified_normalization = normalize_points(sampled)
    assert np.allclose(normalized, condition, atol=1e-6, rtol=0) and np.allclose(verified_normalization, normalization, atol=1e-6, rtol=0), 'Uploaded condition differs from normalized FPS'
    target = condition.astype(np.float64) @ inverse[:3, :3].T+inverse[:3, 3]
    assert np.allclose(target, sampled @ canonical_to_world[:3, :3].T+canonical_to_world[:3, 3], atol=1e-6, rtol=1e-6)
    write_json(folder/'pointcloud-fps-world.json', target.tolist())
    write_json(folder/'pointcloud-condition.json', condition.tolist())
    sampling_check = dict(source='gaussian_filter.ply', points=4096, volumeCandidates=65536,
                          seed=42, opacityThreshold=.2, worldRestorationVerified=True)
    obb = compute_semantic_obb(data['xyz'])
    world_to_bbox = np.eye(4)
    world_to_bbox[:3, :3] = np.asarray(obb['axes'])*1.8/max(obb['size'])
    world_to_bbox[:3, 3] = -world_to_bbox[:3, :3] @ np.asarray(obb['center'])
    P = np.linalg.inv(world_to_bbox)
    frame = dict(obb=obb, pointcloudToWorld=dict(matrix=P.flatten(order='F').tolist()))
    textured = MeshRenderer(source, device, texture_slot)
    raw_points = trimesh.sample.sample_surface(textured.mesh, 8000, seed=0)[0]
    bbox_world, bbox_info = register_inverse_fps(raw_points, target, inverse, np.linalg.inv(P))
    C = np.linalg.solve(P, bbox_world)
    Ci = np.linalg.solve(P, inverse)
    write_json(folder/'stages.json', dict(raw=textured.local.flatten(order='F').tolist(), inverse=(inverse @ textured.local).flatten(order='F').tolist(), bbox=(bbox_world @ textured.local).flatten(order='F').tolist()))
    write_glb(folder/'bbox.glb', textured.doc, textured.binary, bbox_world @ textured.local)
    views, K, framing = fitted_cameras(data, P, np.asarray(textured.mesh.vertices), Ci, C, per_view=per_view)
    textured.set_cameras(views, K)
    print('BBOX', case, 'samples-verified', sample_error, 'camera-distance', framing['cameraDistance'], flush=True)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        textured.render(C, FIT)
        textured.render(C, CHECK, rgb=False, depth=False)
        torch.cuda.current_stream(device).synchronize()
        torch.cuda.empty_cache()
    rgb, ga = gaussian_render(data, frame, views, K, FIT, device, rgb=True)
    _, ga_check = gaussian_render(data, frame, views, K, CHECK, device, rgb=False)
    inverse_rgb, _, _ = textured.render(Ci, FIT, depth=False)
    _, inverse_check, _ = textured.render(Ci, CHECK, rgb=False, depth=False)
    mesh_rgb, mesh_alpha, mesh_depth = textured.render(C, FIT)
    _, bbox_check, _ = textured.render(C, CHECK, rgb=False, depth=False)
    delta, errors, initial_errors, fit, records, counts = fit_images(case, matcher, folder,
        rgb, ga, mesh_rgb, mesh_alpha, mesh_depth, views, K)
    dino = dict(status=fit['status'], reason=fit['reason'])
    for record, error, old_error in zip(records, errors, initial_errors):
        record.update(errorBefore=float(old_error), errorAfter=float(error), inlier=bool(dino['status'] == 'completed' and error < 6))
    final_stage = 'dino' if dino['status'] == 'completed' else 'bbox'
    final = delta @ C
    world = P @ final if final_stage == 'dino' else bbox_world
    final_rgb = mesh_rgb
    final_check = bbox_check
    if final_stage == 'dino':
        final_rgb, _, _ = textured.render(final, FIT, depth=False)
        _, final_check, _ = textured.render(final, CHECK, rgb=False, depth=False)
    validation = dict(inverse=metric(inverse_check, ga_check), bbox=metric(bbox_check, ga_check), final=metric(final_check, ga_check))
    reasons = []
    if final_stage == 'bbox':
        reasons.append('DINO 配准失败，最终结果使用 bbox：'+dino['reason'])
    else:
        if np.sum(np.asarray(fit['inliersPerView']) >= 3) < 2:
            reasons.append('有效内点不足两个视角')
        if fit['conditionNumber'] > 1000:
            reasons.append('求解条件数过大')
        for name in ['inverse', 'bbox']:
            if validation['final']['meanIoU'] < validation[name]['meanIoU']-.005:
                reasons.append(f'共同检查视角平均轮廓比 {name} 下降超过 0.5 个百分点')
    write_glb(folder/'final.glb', textured.doc, textured.binary, world @ textured.local)
    for n, index in enumerate(FIT):
        for name, images in [('gaussian', rgb), ('inverse', inverse_rgb), ('bbox', mesh_rgb), ('final', final_rgb)]:
            write_rgb(folder/f'{name}-{index}.png', images[n])
        panel = np.concatenate([mesh_rgb[n], rgb[n]], axis=1)
        for record in records:
            if record['view'] != index:
                continue
            color = (0, 180, 70) if record['inlier'] else (240, 70, 50)
            a = tuple(np.rint(record['meshPixel']).astype(int))
            b = tuple(np.rint(np.array(record['gaussianPixel'])+[512, 0]).astype(int))
            cv2.line(panel, a, b, color, 1, cv2.LINE_AA)
            cv2.circle(panel, a, 3, color, 1)
            cv2.circle(panel, b, 3, color, 1)
        write_rgb(folder/f'matches-{index}.png', panel)
    report = dict(case=case, status='completed', method='saved inverse -> unnormalized filtered-Gaussian FPS bbox -> eight-view DINOv3/SIFT',
        finalStage=final_stage, dino=dino, framing=framing, matchingVersion=MATCHING_VERSION,
        fitViews=FIT.tolist(), evaluationViews=CHECK.tolist(), rotationFixed=True,
        gaussianCount=len(data['xyz']), bbox=bbox_info, fit=fit, perView=counts, validation=validation,
        diagnosticCheck=dict(accepted=not reasons, reasons=reasons), totalSeconds=time.monotonic()-started,
        stages=dict(raw=textured.local.flatten(order='F').tolist(), inverse=(inverse @ textured.local).flatten(order='F').tolist(), bbox=(bbox_world @ textured.local).flatten(order='F').tolist(), final=(world @ textured.local).flatten(order='F').tolist()),
        relativeScale=float(delta[0, 0]), relativeTranslationCanonical=delta[:3, 3].tolist(),
        frame=frame, samplingVerified=True, samplingMaxError=sample_error, sampling=sampling_check,
        conditionBounds=[condition.min(0).tolist(), condition.max(0).tolist()], fpsWorldBounds=[target.min(0).tolist(), target.max(0).tolist()],
        inputHashes=input_hashes,
        sourceNodeMatrix=textured.local.flatten(order='F').tolist(), embeddedBinarySha256=hashlib.sha256(textured.binary).hexdigest(),
        renderPlan=dict(fitRgbViews=FIT.tolist(), alphaOnlyEvaluationViews=CHECK.tolist(), unusedViews=np.setdiff1d(np.arange(18), np.concatenate([FIT, CHECK])).tolist()))
    report['settings'] = dict(perViewFraming=per_view, textureSlot=texture_slot)
    write_json(folder/'transform.json', dict(matrixOrder='column-major',
        rodinToWorld=dict(matrix=world.flatten(order='F').tolist()), finalStage=final_stage,
        dino=dino))
    write_json(folder/'matches.json', records)
    write_json(folder/'result.json', report)
    print('DONE', case, {k:round(v['meanIoU']*100, 2) for k,v in validation.items()}, reasons, 'seconds', report['totalSeconds'], flush=True)
    return report
