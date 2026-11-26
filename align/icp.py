"""Unchanged numerical functions extracted from the previously used deploy_v4 ICP."""
import time
import numpy as np
from scipy.spatial import ConvexHull,cKDTree

def balanced(points,weights,bins=4):
    """points float64 [N,2/3], weights float64 [N] -> unit-sum float64 [N], equally weighted occupied spatial cells."""
    lo,hi=np.quantile(points,[.01,.99],axis=0)
    cell=np.clip(((points-lo)/np.maximum(hi-lo,1e-6)*bins).astype(int),0,bins-1)
    index=np.ravel_multi_index(cell.T,[bins]*points.shape[1])
    mass=np.bincount(index,weights=weights,minlength=bins**points.shape[1])
    w=weights/np.maximum(mass[index],1e-12)
    return w/w.sum()

def visible_indices(points,eyes):
    """points float64 [N,3], eyes float64 [V,3] world -> visible original indices int64 [M]."""
    radius=100*np.linalg.norm(np.ptp(points,axis=0));indices=set()
    for eye in eyes:
        vectors=points-eye;distance=np.linalg.norm(vectors,axis=1)
        assert np.all(distance>0)
        flipped=vectors*(2*radius/distance[:,None]-1)
        vertices=ConvexHull(np.vstack([flipped,np.zeros((1,3))])).vertices
        indices.update(vertices[vertices<len(points)].tolist())
    selected=np.array(sorted(indices),dtype=np.int64)
    assert len(selected)>=12
    return selected

def icp(source,target,scale_limits):
    """source/target float64 [N/M,3] world, scale_limits [2] -> fixed-rotation scale/translation float64 [4,4], report dict."""
    started=time.monotonic();diagonal=np.linalg.norm(np.ptp(target,axis=0))
    pivot=(target.min(0)+target.max(0))/2
    parameters=np.r_[1.,np.zeros(3)];history=[]
    for level in [.035,.018,.009]:
        _,ai=np.unique(np.floor(source/(diagonal*level)).astype(np.int64),axis=0,return_index=True)
        _,bi=np.unique(np.floor(target/(diagonal*level)).astype(np.int64),axis=0,return_index=True)
        ai,bi=np.sort(ai),np.sort(bi)
        a,b=source[ai],target[bi];tree=cKDTree(b)
        for iteration in range(65):
            moved=a*parameters[0]+parameters[1:]
            distances,nearest=tree.query(moved)
            reverse=cKDTree(moved).query(b)[1]
            mutual=reverse[nearest]==np.arange(len(a))
            within=distances<diagonal*level*3
            keep=mutual&within
            x,y,d=a[keep],b[nearest[keep]],distances[keep]
            if len(x)<12:
                return np.eye(4), dict(status='insufficient_correspondences', inliers=len(x), history=history)
            median=np.median(d);sigma=1.4826*np.median(np.abs(d-median))
            cutoff=max(median+4.685*sigma,diagonal*level)
            robust=np.maximum(0.,1-(d/cutoff)**2)**2
            active=robust>0
            x,y,robust=x[active],y[active],robust[active]
            if len(x)<12:
                return np.eye(4), dict(status='insufficient_correspondences', inliers=len(x), history=history)
            w=balanced(y,robust)
            mx=(x*w[:,None]).sum(0);my=(y*w[:,None]).sum(0)
            scale=float(np.clip(np.sum(w*np.sum((x-mx)*(y-my),axis=1))/np.sum(w*np.sum((x-mx)**2,axis=1)),*scale_limits))
            displacement=np.clip(my-pivot-scale*(mx-pivot),-.15*diagonal,.15*diagonal)
            translation=pivot+displacement-scale*pivot
            update=np.r_[scale,translation]
            change=abs(scale-parameters[0])+np.linalg.norm((scale-parameters[0])*pivot+translation-parameters[1:])/diagonal
            parameters=update
            history.append(dict(level=level,iteration=iteration,scale=scale,mutual=int(mutual.sum()),withinThreshold=int(keep.sum()),inliers=len(x),change=float(change)))
            if change<2e-7:break
    matrix=np.eye(4);matrix[:3,:3]*=parameters[0];matrix[:3,3]=parameters[1:]
    return matrix,dict(status='completed',method='Visible-surface point-to-point ICP with reciprocal correspondences and Tukey weights',
        meshSamples=len(source),targetPoints=len(target),levels=[.035,.018,.009],spatialBins=4,
        fixedRotation=True,fixedScale=False,relativeScale=float(parameters[0]),relativeTranslation=parameters[1:].tolist(),
        rejection='Mutual nearest neighbors; distance < 3*voxel size; Tukey cutoff median+4.685*MAD sigma',
        robustKernel='Tukey',tukeyConstant=4.685,iterations=len(history),converged=bool(change<2e-7),
        finalSourceIndices=ai[keep][active].tolist(),finalTargetIndices=bi[nearest[keep]][active].tolist(),
        finalInlierCount=len(x),history=history,seconds=time.monotonic()-started)
