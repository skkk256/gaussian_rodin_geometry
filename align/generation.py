"""Score every registration candidate and retain the best pose for one Rodin generation."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import trimesh

from gaussian import read_ply
from io_utils import read_rgb, sha256, split_views, write_json
from .geometry import load_glb, register_inverse_fps
from .icp import icp, visible_indices
from .iterative import fit_iterative
from .reference import DINO_ANGLES, FIT, SCORE_VIEWS, reference_cameras
from .rendering import MeshRenderer, cameras, gaussian_render
from .scoring import AlignmentScorer, POLICY
from .routing import FAL_VIEWS, choose_reference

MINIMUM_SCORE = 70.
DINO_SCORE_TOLERANCE = 5.
ALIGNMENT_SETTINGS = dict(version=5, minimumScore=MINIMUM_SCORE, scoring=POLICY,
                          dinoViewIndices=FIT.tolist(),
                          dinoAnglesRadians=DINO_ANGLES.tolist(),
                          referencePolicy=dict(sideViewIndices=[0, 1, 3], sideViewsShareOneRoutingDecision=True,
                                               independentViews=[4, 5], forcedGaussianViews=[2],
                                               driftFeatureSize=512, dinoFeatureSizes=[512, 768], minimumContourPixels=64),
                          scoringAnglesRadians=DINO_ANGLES.tolist(), dinoScoreTolerance=DINO_SCORE_TOLERANCE,
                          icpScaleLimits=[.25, 1.5], maximumRotationDegrees=30,
                          maximumRounds=3, contourStepsPerRound=100,
                          textureSlot='declared-active-texture')


def align_generation(mesh_path: Path, source: Path, work: Path, folder: Path, matcher, sampler, segmenter, device) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    inputs = {name: sha256(work / name) for name in ('canonical.json', 'gaussian.png', 'fal_refine.png', 'gaussian_filter.ply')}
    inputs['mesh'] = sha256(mesh_path)
    inputs['source'] = sha256(source)
    image_only = (work / 'image-only.json').exists()
    if not image_only:
        inputs['condition.npz'] = sha256(work / 'condition.npz')
    report_path = folder / 'alignment.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['inputs'] == inputs and report['settings'] == ALIGNMENT_SETTINGS
        return report
    started = time.perf_counter()
    frame = json.loads((work / 'canonical.json').read_text())
    canonical = np.asarray(frame['canonicalToWorld'], np.float64).reshape(4, 4, order='F')
    data = read_ply(work / 'gaussian_filter.ply')
    inverse = canonical.copy()
    if not image_only:
        with np.load(work / 'condition.npz') as saved:
            inverse = canonical @ np.linalg.inv(saved['normalization'])
            target = saved['canonical'].astype(np.float64) @ canonical[:3, :3].T + canonical[:3, 3]
            restored = saved['points'].astype(np.float64) @ inverse[:3, :3].T + inverse[:3, 3]
            assert np.allclose(target, restored, atol=1e-6, rtol=1e-6)
    elif np.any(data['opacity'] > .2):
        sampled = sampler.sample(data, canonical).astype(np.float64)
        target = sampled @ canonical[:3, :3].T + canonical[:3, 3]
    else:
        report = dict(status='unscorable', reason='Image-only generation has no filtered Gaussian with opacity > 0.2',
                      score=None, finalStage='inverse', matrix=inverse.flatten(order='F').tolist(),
                      inputs=inputs, settings=ALIGNMENT_SETTINGS, stages={}, imageOnly=True)
        write_json(report_path, report)
        return report
    np.savez_compressed(folder / 'reference-fps.npz', world=target)
    views, K, P = reference_cameras(frame, DINO_ANGLES)
    render_frame = dict(pointcloudToWorld=dict(matrix=P.flatten(order='F').tolist()))
    doc, _ = load_glb(mesh_path)
    material = doc['materials'][doc['meshes'][0]['primitives'][0]['material']]
    slot = 'emissive' if 'emissiveTexture' in material and max(material.get('emissiveFactor', [0, 0, 0])) > 0 else 'base-color'
    mesh = MeshRenderer(mesh_path, device, slot)
    mesh.set_cameras(views, K)
    raw_points = trimesh.sample.sample_surface(mesh.mesh, 8000, seed=0)[0]
    bbox, bbox_info = register_inverse_fps(raw_points, target, inverse, np.linalg.inv(P))
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        mesh.render(np.linalg.solve(P, bbox), FIT)
        torch.cuda.current_stream(device).synchronize()
        torch.cuda.empty_cache()
    reference_cache = work / 'gaussian-reference.npz'
    if not reference_cache.exists():
        original_rgb, original_alpha = gaussian_render(read_ply(source), render_frame, views, K, FIT, device, rgb=True)
        _, filtered_alpha = gaussian_render(data, render_frame, views, K, SCORE_VIEWS, device, rgb=False)
        reference_alpha = original_alpha * np.stack([
            cv2.dilate((a > .15).astype(np.uint8), np.ones((101, 101), np.uint8)) for a in filtered_alpha])
        contour = np.stack([cv2.erode((a > .5).astype(np.float32), np.ones((3, 3), np.uint8)) for a in filtered_alpha])
        np.savez_compressed(reference_cache, rgb=original_rgb, alpha=reference_alpha, contour=contour,
                            original_alpha=original_alpha, filtered_alpha=filtered_alpha, views=views, K=K)
    with np.load(reference_cache) as saved:
        reference_rgb, reference_alpha, contour, original_alpha, filtered_alpha = (
            saved[name] for name in ('rgb', 'alpha', 'contour', 'original_alpha', 'filtered_alpha'))
    mixed_cache = work / 'alignment-reference.npz'
    if not mixed_cache.exists():
        fal = np.stack(split_views(read_rgb(work / 'fal_refine.png')))
        fal_alpha = np.zeros_like(original_alpha)
        for i in FAL_VIEWS:
            fal_alpha[i] = np.maximum(segmenter.predict(fal[i]), segmenter.predict_detail(fal[i]))
        scan = np.stack(split_views(read_rgb(work / 'gaussian.png')))
        routing = choose_reference(matcher, scan, fal, fal_alpha)
        for i, source_name in enumerate(routing['sources']):
            if source_name == 'FAL':
                reference_rgb[i] = fal[i]
                reference_alpha[i] = fal_alpha[i]
                contour[i] = fal_alpha[i]
        reference_alpha[(contour > .5).sum((1, 2)) < 64] = 0
        routing['activeViews'] = np.flatnonzero((contour > .5).sum((1, 2)) >= 64).tolist()
        routing['pixelCounts'] = (contour > .5).sum((1, 2)).tolist()
        write_json(work / 'reference-routing.json', routing)
        np.savez_compressed(mixed_cache, rgb=reference_rgb, alpha=reference_alpha, contour=contour,
                            sources=np.asarray(routing['sources']))
    routing = json.loads((work / 'reference-routing.json').read_text())
    with np.load(mixed_cache) as saved:
        reference_rgb, reference_alpha, contour = (saved[name] for name in ('rgb', 'alpha', 'contour'))
        assert saved['sources'].tolist() == routing['sources']
    assert routing['sources'][2] == 'Gaussian'
    assert len(reference_rgb) == len(reference_alpha) == len(contour) == len(original_alpha) == len(FIT)
    assert len(filtered_alpha) == len(SCORE_VIEWS)
    scorer = AlignmentScorer(target, device)
    stages = {}
    case = f'{source.stem}/attempt-{folder.name}'

    def evaluate(name, matrix):
        """External Rodin-to-world float64 [4,4] -> weighted score; record every candidate without baking its transform."""
        _, alpha, _ = mesh.render(np.linalg.solve(P, matrix), SCORE_VIEWS, rgb=False, depth=False)
        metrics = scorer.evaluate(mesh.mesh, matrix, filtered_alpha, alpha)
        stages[name] = dict(matrix=matrix.flatten(order='F').tolist(), metrics=metrics)
        print(f'ALIGN {case} {name}: {metrics["score"]:.2f}', flush=True)
        return metrics['score']

    best_stage, best_world = 'bbox', bbox
    best_score = evaluate('bbox', bbox)
    icp_info = dict(status='skipped', reason='bbox score >= 70')
    if best_score < MINIMUM_SCORE:
        world_to_camera_frame = np.linalg.inv(P)
        canonical_target = target @ world_to_camera_frame[:3, :3].T + world_to_camera_frame[:3, 3]
        hpr_views, _ = cameras()
        eyes = -np.einsum('vji,vj->vi', hpr_views[:, :3, :3], hpr_views[:, :3, 3])
        surface = canonical_target[visible_indices(canonical_target, eyes)]
        points = trimesh.sample.sample_surface(mesh.mesh, 16384, seed=42)[0]
        start = np.linalg.solve(P, bbox)
        delta, icp_info = icp(points @ start[:3, :3].T + start[:3, 3], surface, (.25, 1.5))
        if icp_info['status'] == 'completed':
            candidate = P @ delta @ start
            score = evaluate('icp', candidate)
            if score > best_score:
                best_score, best_stage, best_world = score, 'icp', candidate
    dino_start = best_stage
    dino_start_score = best_score
    C = np.linalg.solve(P, best_world)
    mesh_rgb, mesh_alpha, mesh_depth = mesh.render(C, FIT)
    with torch.cuda.device(device):
        delta, errors, initial_errors, fit, records, counts = fit_iterative(case, matcher, folder,
            reference_rgb, reference_alpha, mesh_rgb, mesh_alpha, mesh_depth, views, K, mesh, C,
            contour=contour, original_alpha=original_alpha, source=routing['sources'],
            view_weights=np.ones(len(FIT)))
    candidate = P @ delta @ C
    score = evaluate('dino', candidate)
    if score >= best_score - DINO_SCORE_TOLERANCE:
        best_score, best_stage, best_world = score, 'dino', candidate
    for record, before, after in zip(records, initial_errors, errors, strict=True):
        record.update(errorBefore=float(before), errorAfter=float(after), inlier=bool(after < 6))
    write_json(folder / 'matches.json', records)
    report = dict(status='completed', score=best_score, accepted=best_score >= MINIMUM_SCORE,
                  finalStage=best_stage, matrix=best_world.flatten(order='F').tolist(),
                  dinoStart=dino_start, dinoStartScore=dino_start_score,
                  dinoImproved=score > dino_start_score, dinoSelected=best_stage == 'dino',
                  dinoScoreDrop=dino_start_score - score, stages=stages,
                  bbox=bbox_info, icp=icp_info, dino=fit, perView=counts, reference=routing,
                  inputs=inputs, settings=ALIGNMENT_SETTINGS, textureSlot=slot, imageOnly=image_only,
                  seconds=time.perf_counter() - started)
    write_json(report_path, report)
    return report
