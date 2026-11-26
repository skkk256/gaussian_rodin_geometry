"""Sparse-view DINO/SIFT correspondences and explicit consensus outcomes."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import DINOv3ViTModel
from transformers.utils import logging

RESOLUTION = 512
MATCHING_VERSION = "sparse-views-consensus-v4"


def clarity(rgb,alpha):
    """rgb uint8 [H,W,3], alpha float32 [H,W] -> detail/clarity float32 [H,W] and gray uint8 [H,W]."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    image=gray.astype(np.float32)/255
    fine=cv2.GaussianBlur(image,(0,0),.7)
    middle=cv2.GaussianBlur(image,(0,0),2.)
    broad=cv2.GaussianBlur(image,(0,0),5.)
    hf=np.sqrt(np.maximum(cv2.boxFilter((fine-middle)**2,-1,(11,11)),0))
    lf=np.sqrt(np.maximum(cv2.boxFilter((fine-broad)**2,-1,(11,11)),0))
    detail=np.clip((hf/(lf+.005)-.08)/.6,0,1)*np.clip(lf/.025,0,1)*np.clip(alpha,0,1)
    assert np.isfinite(detail).all()
    return detail.astype(np.float32),gray

def at(image,points):
    """image float32 [H,W], points float32 [N,2] -> sampled float32 [N]."""
    return cv2.remap(image,points[:,0].astype(np.float32)[None],points[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]


def projection(points, parameters, views, K, ids):
    """points float64 [N,3], parameters float64 [4], cameras float32 [V,4,4]/[V,3,3], ids int64 [N] -> pixels float64 [N,2]."""
    xyz=points*parameters[0]+parameters[1:]
    camera=np.einsum('nij,nj->ni',views[ids,:3,:3],xyz)+views[ids,:3,3]
    return camera[:,:2]/camera[:,2:]*K[ids][:,[0,1],[0,1]]+K[ids,:2,2]


class DinoMatcher:
    def __init__(self, weights: Path, device: torch.device):
        self.device = device
        assert device.type == 'cuda' and device.index is not None and weights.is_dir()
        logging.disable_progress_bar()
        with torch.cuda.device(device):
            self.model = DINOv3ViTModel.from_pretrained(str(weights), local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
            assert self.model.config.patch_size == 16
            torch.cuda.empty_cache()
            for size in (512, 768):
                self.model(pixel_values=torch.zeros((1, 3, size, size), dtype=torch.bfloat16, device=device))
            torch.cuda.current_stream(device).synchronize()
            torch.cuda.empty_cache()

    def pyramid(self, images: np.ndarray) -> list[torch.Tensor]:
        """RGB uint8 [8,512,512,3] -> two feature grids float32 [8,C,32/48,32/48] on owning CUDA device."""
        outputs = []
        with torch.cuda.device(self.device):
            mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
            std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
            for size in (512, 768):
                features = []
                for image in images:
                    pixels = torch.as_tensor(image, device=self.device).permute(2, 0, 1)[None].float()/255
                    pixels = F.interpolate(pixels, size=(size, size), mode='bilinear', align_corners=False)
                    tokens = self.model(pixel_values=((pixels-mean)/std).to(torch.bfloat16)).last_hidden_state
                    patches = tokens[:, 1+self.model.config.num_register_tokens:]
                    assert patches.shape[1] == (size//16)**2
                    features.append(F.normalize(patches.float(), dim=-1).transpose(1, 2).reshape(1, -1, size//16, size//16))
                outputs.append(torch.cat(features))
        return outputs

    def descriptors(self, features: list[torch.Tensor], index: int, points: np.ndarray) -> np.ndarray:
        """Feature grids float32 [8,C,G,G], pixels float32 [N,2] -> descriptors float32 [N,2C]."""
        with torch.cuda.device(self.device):
            assert all(f.device == self.device for f in features)
            grid = torch.as_tensor((points+.5)/RESOLUTION*2-1, dtype=torch.float32, device=self.device)[None, None]
            sampled = [F.normalize(F.grid_sample(f[index:index+1], grid, align_corners=False)[0, :, 0].T, dim=-1) for f in features]
            return (torch.cat(sampled, dim=1)/np.sqrt(2)).cpu().numpy()



@dataclass
class ViewMatches:
    mesh_pixels: np.ndarray
    gaussian_pixels: np.ndarray
    weights: np.ndarray
    records: list
    info: dict


@dataclass
class ReprojectionFit:
    status: str
    reason: str
    parameters: np.ndarray | None
    errors: np.ndarray
    statistics: dict


def corners(gray, alpha, quality, depth):
    """gray uint8 [512,512], alpha/quality float32 [512,512], optional depth float32 [512,512] -> pixels float32 [N,2], SIFT float32 [N,128], counts; N may be zero."""
    detected = cv2.goodFeaturesToTrack(gray, maxCorners=450, qualityLevel=.008, minDistance=7,
        blockSize=7, mask=((alpha > .5)*255).astype(np.uint8))
    counts = dict(detected=0, clear=0, onSurface=0)
    if detected is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32), counts
    xy = detected[:, 0]
    counts['detected'] = len(xy)
    xy = xy[at(quality, xy) > .12]
    counts['clear'] = len(xy)
    if depth is not None and len(xy):
        offsets = np.array([(a, b) for a in range(-2, 3) for b in range(-2, 3)])
        pixels = np.clip(np.rint(xy[:, None]+offsets).astype(int), 0, 511)
        good = (depth[pixels[..., 1], pixels[..., 0]] > 0)&(alpha[pixels[..., 1], pixels[..., 0]] > .9)
        distance = np.linalg.norm(pixels-xy[:, None], axis=-1)
        distance[~good] = 1e6
        keep = distance.min(1) < 3
        xy = pixels[np.arange(len(xy)), distance.argmin(1)][keep].astype(np.float32)
    counts['onSurface'] = len(xy)
    if not len(xy):
        return xy, np.empty((0, 128), np.float32), counts
    keypoints = [cv2.KeyPoint(float(x), float(y), 24, 0) for x, y in xy]
    _, sift = cv2.SIFT_create().compute(gray, keypoints)
    sift = sift.astype(np.float32)
    sift /= np.linalg.norm(sift, axis=1, keepdims=True).clip(1e-8)
    return xy, sift, counts


def match_view(matcher, features, ordinal, view_id, gs, mesh, ga, ma, depth):
    """RGB uint8 [512,512,3], alpha/depth float32 [512,512] -> ViewMatches with float32 [N,2] pixels, float64 [N] weights, including empty views."""
    gq, gg = clarity(gs, ga)
    mq, mg = clarity(mesh, ma)
    a, sa, mesh_counts = corners(mg, ma, mq, depth)
    b, sb, gaussian_counts = corners(gg, ga, gq, None)
    info = dict(view=int(view_id), meshCorners=len(a), gaussianCorners=len(b), matches=0,
        meshCornerCounts=mesh_counts, gaussianCornerCounts=gaussian_counts, status='no_features', reason='')
    if not len(a) or not len(b):
        info['reason'] = f'可用角点不足：网格 {len(a)}，高斯 {len(b)}；该视角不参与求解'
        return ViewMatches(np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty(0), [], info)
    da = matcher.descriptors(features, ordinal+features[0].shape[0]//2, a)
    db = matcher.descriptors(features, ordinal, b)
    cosine, sift = da @ db.T, sa @ sb.T
    distance = np.linalg.norm(a[:, None]-b[None], axis=-1)
    score = .7*cosine+.3*sift-.025*(distance/64)**2
    score[(distance > 80)|(cosine < .5)|(sift < .25)] = -2
    forward, reverse = score.argmax(1), score.argmax(0)
    pairs, weights, records = [], [], []
    for i, j in enumerate(forward):
        if reverse[j] != i or score[i, j] < .45:
            continue
        remote = np.linalg.norm(b-b[j], axis=1) > 18
        if not remote.any():
            continue
        margin = score[i, j]-score[i, remote].max()
        if margin < .01:
            continue
        weight = np.clip(margin/.08, .15, 2)*np.clip(at(gq, b[j:j+1])[0], .1, 1)*np.clip((cosine[i, j]-.5)/.4, .03, 1)**2
        pairs.append((i, j))
        weights.append(weight)
        records.append(dict(view=int(view_id), meshPixel=a[i].tolist(), gaussianPixel=b[j].tolist(),
            dinoCosine=float(cosine[i, j]), siftCosine=float(sift[i, j]), margin=float(margin), weight=float(weight)))
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    info.update(matches=len(pairs), status='matched' if len(pairs) else 'no_matches',
        reason='' if len(pairs) else '角点存在，但没有通过双向、相似度和歧义筛选的对应；该视角不参与求解')
    return ViewMatches(a[pairs[:, 0]], b[pairs[:, 1]], np.asarray(weights), records, info)


def fit_reprojection(points, targets, ids, weights, views, K):
    """points float64 [N,3], targets float64 [N,2], ids int64 [N], weights float64 [N], views/K float32 [V,4,4]/[V,3,3] -> ReprojectionFit; insufficient evidence is an explicit outcome."""
    assert points.shape == (len(ids), 3) and targets.shape == (len(ids), 2) and weights.shape == ids.shape
    assert np.isfinite(points).all() and np.isfinite(targets).all() and np.isfinite(weights).all() and np.all(weights > 0)
    initial = np.array([1., 0., 0., 0.])
    initial_errors = np.linalg.norm(projection(points, initial, views, K, ids)-targets, axis=1)
    counts = np.bincount(ids, minlength=len(views))
    statistics = dict(matches=len(points), inliers=0, matchesPerView=counts.tolist(),
        inliersPerView=[0]*len(views), inlierThresholdPixels=6, requiredMatches=8, requiredInliers=6,
        requiredViews=2, maximumConsensus=0, admissibleCandidates=0)
    if len(points) < 8 or np.count_nonzero(counts) < 2:
        return ReprojectionFit('insufficient_matches', f'共有 {len(points)} 对候选，覆盖 {np.count_nonzero(counts)} 个视角；至少需要 8 对候选和 2 个视角', None, initial_errors, statistics)
    R, t = views[ids, :3, :3], views[ids, :3, 3]
    uv = (targets-K[ids, :2, 2])/K[ids][:, [0, 1], [0, 1]]
    equations = R[:, :2]-uv[:, :, None]*R[:, 2:3]
    A = np.concatenate([np.einsum('nkj,nj->nk', equations, points)[..., None], equations], axis=-1)
    rhs = uv*t[:, 2:]-t[:, :2]
    rng = np.random.default_rng(20261120)
    chosen, best_score = None, -1.
    for _ in range(600):
        sample = rng.choice(len(points), 6, replace=False, p=weights/weights.sum())
        if len(np.unique(ids[sample])) < 2:
            continue
        candidate, _, rank, _ = np.linalg.lstsq(A[sample].reshape(-1, 4), rhs[sample].ravel(), rcond=None)
        if rank < 4 or not (.8 < candidate[0] < 1.2 and np.max(np.abs(candidate[1:])) < .15):
            continue
        statistics['admissibleCandidates'] += 1
        errors = np.linalg.norm(projection(points, candidate, views, K, ids)-targets, axis=1)
        good = errors < 6
        statistics['maximumConsensus'] = max(statistics['maximumConsensus'], int(good.sum()))
        if good.sum() < 6 or len(np.unique(ids[good])) < 2:
            continue
        score = np.sum(np.maximum(0, 1-(errors/5)**2)*weights/np.sqrt(counts[ids]))
        if score > best_score:
            chosen, best_score = candidate, score
    if chosen is None:
        reason = f"共 {len(points)} 对候选；600 次 RANSAC 最大共识 {statistics['maximumConsensus']} 对，需要至少 6 对内点并覆盖 2 个视角"
        return ReprojectionFit('insufficient_consensus', reason, None, initial_errors, statistics)
    for _ in range(4):
        errors = np.linalg.norm(projection(points, chosen, views, K, ids)-targets, axis=1)
        good = errors < 6
        def residual(parameters):
            """parameters float64 [4] -> weighted reprojection residual float64 [2*M]."""
            return ((projection(points[good], parameters, views, K, ids[good])-targets[good])*np.sqrt(weights[good, None])).ravel()
        refined = least_squares(residual, chosen, bounds=([.8, -.15, -.15, -.15], [1.2, .15, .15, .15]),
            loss='cauchy', f_scale=1.5, max_nfev=150)
        new_errors = np.linalg.norm(projection(points, refined.x, views, K, ids)-targets, axis=1)
        new_good = new_errors < 6
        new_score = np.sum(np.maximum(0, 1-(new_errors/5)**2)*weights/np.sqrt(counts[ids]))
        if new_good.sum() < 6 or len(np.unique(ids[new_good])) < 2 or new_score < best_score:
            statistics['refinementStoppedReason'] = '精化降低共识，保留有效 RANSAC 解'
            break
        chosen, best_score = refined.x, new_score
    errors = np.linalg.norm(projection(points, chosen, views, K, ids)-targets, axis=1)
    good = errors < 6
    camera = np.einsum('nij,nj->ni', R, points*chosen[0]+chosen[1:])+t
    derivative = np.concatenate([np.einsum('nij,nj->ni', R, points)[..., None], R], axis=-1)
    jacobian = (derivative[:, :2]*camera[:, 2:, None]-camera[:, :2, None]*derivative[:, 2:])/camera[:, 2:, None]**2
    jacobian *= K[ids][:, [0, 1], [0, 1], None]*np.sqrt(weights[:, None, None])
    singular = np.linalg.svd(jacobian[good].reshape(-1, 4), compute_uv=False)
    if singular[-1] <= singular[0]*1e-10:
        return ReprojectionFit('degenerate_geometry', '匹配的几何约束不足以确定统一尺度和平移', None, initial_errors, statistics)
    statistics.update(inliers=int(good.sum()), inliersPerView=np.bincount(ids[good], minlength=len(views)).tolist(),
        medianReprojectionPixels=float(np.median(errors)), inlierMedianPixels=float(np.median(errors[good])),
        conditionNumber=float(singular[0]/singular[-1]), medianPixelsBefore=float(np.median(initial_errors)),
        inlierMedianPixelsBefore=float(np.median(initial_errors[good])), inlierFraction=float(good.mean()))
    return ReprojectionFit('completed', '', chosen, errors, statistics)
