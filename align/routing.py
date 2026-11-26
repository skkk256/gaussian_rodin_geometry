"""Category-independent checks for restoration drift relative to the original render."""
import time
import cv2
import numpy as np
import torch


FAL_VIEWS = np.array([0, 1, 3, 4, 5])


def choose_reference(matcher, original_rgb, fal, alpha):
    """RGB uint8 [6,512,512,3], FAL alpha float32 [6,512,512] -> per-view source and drift report; back is always Gaussian."""
    started = time.perf_counter()
    features = matcher.pyramid(np.concatenate([original_rgb[FAL_VIEWS], fal[FAL_VIEWS]]))[0]
    grid = np.stack(np.meshgrid(np.arange(32)*16+7.5, np.arange(32)*16+7.5), axis=-1).reshape(-1, 2)
    reports = []
    for index, i in enumerate(FAL_VIEWS):
        roi = cv2.dilate((alpha[i] > .5).astype(np.uint8), np.ones((101, 101), np.uint8))
        keep = alpha[i, grid[:, 1].astype(int), grid[:, 0].astype(int)] > .9
        keep_target = roi[grid[:, 1].astype(int), grid[:, 0].astype(int)] > 0
        source, target = grid[keep], grid[keep_target]
        a = features[index+len(FAL_VIEWS)].flatten(1).T[torch.as_tensor(keep, device=matcher.device)]
        b = features[index].flatten(1).T[torch.as_tensor(keep_target, device=matcher.device)]
        row = dict(view=int(i), status='insufficient_correspondences', matches=0, reliable=False, reasons=[])
        if min(len(source), len(target)) < 6:
            row['reasons'].append('fewer than six foreground DINO cells')
            reports.append(row)
            continue
        cosine = (a @ b.T).cpu().numpy()
        distance = np.linalg.norm(source[:, None]-target[None], axis=-1)
        score = cosine-.01*(distance/64)**2
        score[distance > 100] = -2
        forward, reverse = score.argmax(1), score.argmax(0)
        pairs = np.asarray([(j, k) for j, k in enumerate(forward) if reverse[k] == j and cosine[j, k] > .55], dtype=int).reshape(-1, 2)
        source, target = source[pairs[:, 0]], target[pairs[:, 1]]
        row['matches'] = len(source)
        if len(source) < 6:
            row['reasons'].append('fewer than six mutual DINO correspondences')
            reports.append(row)
            continue
        M, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
            ransacReprojThreshold=8, maxIters=10000, confidence=.999, refineIters=30)
        if M is None:
            row.update(status='no_2d_consensus', reasons=['no robust 2D similarity consensus'])
            reports.append(row)
            continue
        good = inliers[:, 0].astype(bool)
        scale = np.linalg.norm(M[:, 0])
        angle = np.rad2deg(np.arctan2(M[1, 0], M[0, 0]))
        shift = M @ np.array([256, 256, 1])-256
        coverage = cv2.contourArea(cv2.convexHull(source[good].astype(np.float32)))/max(float((alpha[i] > .5).sum()), 1.)
        tests = [('fewer than 20 inliers', good.sum() >= 20), ('inlier fraction below 35%', good.mean() >= .35),
            ('inliers cover less than 12% of foreground', coverage >= .12), ('scale drift exceeds 10%', .9 <= scale <= 1.1),
            ('2D rotation drift exceeds 5 degrees', abs(angle) <= 5), ('image-center displacement exceeds 20 pixels', np.linalg.norm(shift) <= 20)]
        row.update(status='measured', inliers=int(good.sum()), inlierFraction=float(good.mean()), spatialCoverage=float(coverage),
            scale=float(scale), rotationDegrees=float(angle), centerShiftPixels=shift.tolist(), matrix=M.tolist(),
            reliable=all(passed for _, passed in tests), reasons=[name for name, passed in tests if not passed])
        reports.append(row)
    side_fal = all(row['reliable'] for row in reports[:3])
    sources = ['FAL' if side_fal else 'Gaussian', 'FAL' if side_fal else 'Gaussian',
               'Gaussian', 'FAL' if side_fal else 'Gaussian']
    sources += ['FAL' if row['reliable'] else 'Gaussian' for row in reports[3:]]
    reports.insert(2, dict(view=2, status='forced_gaussian', source='Gaussian',
                           reason='Back uses a 30-degree Gaussian render; the 0-degree FAL back is excluded'))
    for row, source in zip(reports, sources, strict=True):
        row['source'] = source
    return dict(sources=sources, views=reports, seconds=time.perf_counter()-started,
        sideViewsShareOneRoutingDecision=True, sideViewIndices=[0, 1, 3], forcedGaussianViews=[2],
        extraViewsIndividuallyRequireDistributedFalConsensus=True,
        policy='Front/right/left jointly require reliable FAL drift checks; top/bottom independently require them; back is Gaussian',
        gaussianImagesUsedForMatching='Gaussian' in sources, gaussianImagesUsedForContours='Gaussian' in sources,
        correction='none; DINO grid measures drift but does not warp images')
