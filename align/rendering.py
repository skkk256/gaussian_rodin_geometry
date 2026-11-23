"""Shared mesh/Gaussian cameras, conservative framing and GPU rendering."""
import cv2
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

from .geometry import accessor, load_glb, node_matrix

FIT = np.array([0, 2, 4, 6, 8, 10, 12, 14])
CHECK = np.array([1, 5, 9, 13])
RESOLUTION = 512
RENDER_VERSION = "fit-bounds-bbox-on-dino-failure-v2"


def cameras() -> tuple[np.ndarray, np.ndarray]:
    """Return OpenCV canonical views float32 [18,4,4] and intrinsics float32 [18,3,3]."""
    angles = np.deg2rad(np.tile(np.arange(0, 360, 45), 2))
    elevation = np.deg2rad(np.repeat([30., -30.], 8))
    eyes = 5 * np.column_stack([np.cos(elevation)*np.sin(angles), np.sin(elevation), np.cos(elevation)*np.cos(angles)])
    eyes = np.concatenate([eyes, [[0., 5., 0.], [0., -5., 0.]]])
    views = np.tile(np.eye(4), (18, 1, 1))
    for i, eye in enumerate(eyes):
        forward = -eye / np.linalg.norm(eye)
        up = np.array([0., 1., 0.]) if i < 16 else np.array([0., 0., 1.])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        views[i, :3, :3] = np.stack([right, -np.cross(right, forward), forward])
        views[i, :3, 3] = -views[i, :3, :3] @ eye
    focal = RESOLUTION * 2.5 / (2 * .58)
    intrinsics = np.tile([[focal, 0, RESOLUTION/2], [0, focal, RESOLUTION/2], [0, 0, 1]], (18, 1, 1))
    return views.astype(np.float32), intrinsics.astype(np.float32)


class MeshRenderer:
    def __init__(self, path, device, texture_slot):
        self.doc, self.binary = load_glb(path)
        d = self.doc
        assert len(d['nodes']) == len(d['meshes']) == len(d['meshes'][0]['primitives']) == 1
        primitive = d['meshes'][0]['primitives'][0]
        attributes = primitive['attributes']
        self.mesh = trimesh.Trimesh(vertices=accessor(d, self.binary, attributes['POSITION']),
            faces=accessor(d, self.binary, primitive['indices']).reshape(-1, 3), process=False)
        self.local = node_matrix(d['nodes'][0])
        self.mesh.apply_transform(self.local)
        material = d['materials'][primitive['material']]
        texture_info = (material['emissiveTexture'] if texture_slot == 'emissive'
                        else material['pbrMetallicRoughness']['baseColorTexture'])
        assert texture_info.get('texCoord', 0) == 0 and not texture_info.get('extensions'), 'Texture must use untransformed TEXCOORD_0'
        texture_index = texture_info['index']
        image = d['images'][d['textures'][texture_index]['source']]
        view = d['bufferViews'][image['bufferView']]
        offset = view.get('byteOffset', 0)
        decoded = cv2.imdecode(np.frombuffer(self.binary[offset:offset+view['byteLength']], np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None, 'Cannot decode the embedded RGB texture'
        texture = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        factor = material.get('emissiveFactor', [0., 0., 0.]) if texture_slot == 'emissive' else material['pbrMetallicRoughness'].get('baseColorFactor', [1., 1., 1., 1.])[:3]
        texture = np.clip(np.rint(texture.astype(np.float32)*factor), 0, 255).astype(np.uint8)
        self.device = device
        with torch.cuda.device(device):
            self.context = dr.RasterizeCudaContext(device=device)
            self.vertices = torch.as_tensor(np.asarray(self.mesh.vertices), dtype=torch.float32, device=device)
            self.faces = torch.as_tensor(np.asarray(self.mesh.faces), dtype=torch.int32, device=device).contiguous()
            self.uv = torch.as_tensor(accessor(d, self.binary, attributes['TEXCOORD_0']), dtype=torch.float32, device=device)
            self.texture = torch.as_tensor(texture, dtype=torch.float32, device=device)[None]/255

    def set_cameras(self, views, intrinsics):
        """views float32 [18,4,4], intrinsics float32 [18,3,3] -> owning-device camera tensors."""
        with torch.cuda.device(self.device):
            self.views = torch.as_tensor(views, dtype=torch.float32, device=self.device)
            self.intrinsics = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)

    def render(self, matrix, indices, *, rgb=True, depth=True):
        """matrix float64 [4,4], indices int64 [V] -> optional RGB uint8 [V,512,512,3], alpha float32 [V,512,512], optional depth float32 [V,512,512]."""
        with torch.cuda.device(self.device):
            assert self.texture.device == self.uv.device == self.vertices.device == self.faces.device == self.views.device == self.intrinsics.device == self.device
            m = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
            xyz = self.vertices @ m[:3, :3].T+m[:3, 3]
            views, k = self.views[indices], self.intrinsics[indices]
            camera = torch.einsum('vij,nj->vni', views[:, :3, :3], xyz)+views[:, None, :3, 3]
            z = camera[..., 2]
            clip = torch.stack([2*k[:, 0, 0, None]*camera[..., 0]/512,
                -2*k[:, 1, 1, None]*camera[..., 1]/512, (40.02/39.98)*z-1.6/39.98, z], dim=-1).contiguous()
            raster, _ = dr.rasterize(self.context, clip, self.faces, (512, 512))
            occupied = raster[..., 3:] > 0
            alpha = dr.antialias(occupied.to(torch.float32), raster, clip, self.faces)[..., 0].flip(1)
            color = None
            if rgb:
                uv = dr.interpolate(self.uv[None].expand(len(indices), -1, -1).contiguous(), raster, self.faces)[0]
                color = dr.texture(self.texture, uv, filter_mode='linear', boundary_mode='wrap')
                color = dr.antialias(torch.where(occupied, color, 1.), raster, clip, self.faces).flip(1)
                color = (color.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy()
            distance = dr.interpolate(camera[..., 2:].contiguous(), raster, self.faces)[0][..., 0].flip(1).cpu().numpy() if depth else None
            return color, alpha.cpu().numpy(), distance


def fitted_cameras(data, P, vertices, inverse, bbox, *, per_view):
    """Gaussian world arrays [N,...], P/inverse/bbox float64 [4,4], vertices float64 [M,3] -> views float32 [18,4,4], K float32 [18,3,3], framing metadata."""
    from scipy.spatial.transform import Rotation
    canonical = np.linalg.inv(P)
    means = data['xyz'] @ canonical[:3, :3].T+canonical[:3, 3]
    rotations = Rotation.from_quat(data['rotation'][:, [1, 2, 3, 0]]).as_matrix()
    basis = np.einsum('ij,njk->nik', canonical[:3, :3], rotations)*data['scale'][:, None, :]
    inverse_points = vertices @ inverse[:3, :3].T+inverse[:3, 3]
    bbox_points = vertices @ bbox[:3, :3].T+bbox[:3, 3]
    lower = np.minimum.reduce([means.min(0), inverse_points.min(0), bbox_points.min(0)])
    upper = np.maximum.reduce([means.max(0), inverse_points.max(0), bbox_points.max(0)])
    center = (lower+upper)/2
    views, intrinsics = cameras()
    margin = 28
    required, near_required = [], []
    for view, k in zip(views, intrinsics):
        depth_direction = -view[2, :3]
        near_required.append(max(((means-center) @ depth_direction+3*np.linalg.norm(np.einsum('i,nij->nj', depth_direction, basis), axis=1)).max(),
            ((inverse_points-center) @ depth_direction).max(),
            (np.maximum(.8*(bbox_points @ depth_direction), 1.2*(bbox_points @ depth_direction))-center @ depth_direction+.15*np.abs(depth_direction).sum()).max())+.05)
        for axis in (0, 1):
            for sign in (-1, 1):
                plane = sign*k[axis, axis]/(256-margin)*view[axis, :3]-view[2, :3]
                gaussian_support = (means-center) @ plane+3*np.linalg.norm(np.einsum('i,nij->nj', plane, basis), axis=1)
                inverse_support = (inverse_points-center) @ plane
                bbox_projection = bbox_points @ plane
                optimizer_support = np.maximum(.8*bbox_projection, 1.2*bbox_projection)-center @ plane+.15*np.abs(plane).sum()
                required.append(max(gaussian_support.max(), inverse_support.max(), optimizer_support.max()))
    required = np.asarray(required).reshape(18, 4).max(1)
    distances = np.maximum(required, near_required)+1e-4
    if not per_view:
        distances[:] = max(5., float(distances.max()))
    assert np.max(distances) + max(np.linalg.norm(means-center, axis=1).max(), np.linalg.norm(inverse_points-center, axis=1).max(), 1.2*np.linalg.norm(bbox_points, axis=1).max()+np.linalg.norm(center)+.15*np.sqrt(3)) < 40, 'Object exceeds the renderer far plane'
    views[:, :3, 3] = -np.einsum('vij,j->vi', views[:, :3, :3], center)
    views[:, 2, 3] += distances
    assert np.isfinite(distances).all() and np.all(distances >= required)
    version = 'fit-per-view-bounds-v3' if per_view else RENDER_VERSION
    return views, intrinsics, dict(version=version, marginPixels=margin, cameraDistance=float(distances.max()),
        cameraDistances=distances.tolist(), perViewFraming=per_view,
        centerCanonical=center.tolist(), boundsCanonical=[lower.tolist(), upper.tolist()],
        minimumFrustumClearanceCanonical=float(np.min(distances-required)),
        views=views.tolist(), intrinsics=intrinsics.tolist(), gaussianSupportSigma=3,
        includesInverse=True, includesBbox=True, includesOptimizerLimits=True)


def gaussian_render(data, frame, views, intrinsics, indices, device, *, rgb):
    """Gaussian world arrays [N,...], cameras float32 [18,4,4]/[18,3,3], indices int64 [V] -> optional RGB uint8 [V,512,512,3], alpha float32 [V,512,512]."""
    from gsplat import rasterization
    matrix = np.asarray(frame['pointcloudToWorld']['matrix']).reshape(4, 4, order='F')
    scale = np.linalg.norm(matrix[:3, 0])
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3)*scale**2, atol=1e-8)
    world_views = np.tile(np.eye(4), (len(indices), 1, 1))
    world_views[:, :3, :3] = views[indices, :3, :3] @ (matrix[:3, :3]/scale).T
    world_views[:, :3, 3] = scale*views[indices, :3, 3]-np.einsum('vij,j->vi', world_views[:, :3, :3], matrix[:3, 3])
    records = data['records']
    names = sorted((k for k in records.dtype.names if k.startswith('f_rest_')), key=lambda k: int(k[7:]))
    assert len(names) in (0, 45)
    count = len(names)//3
    dc = np.column_stack([records[f'f_dc_{i}'] for i in range(3)])[:, None]
    rest = np.array([records[k] for k in names], dtype=np.float32).T.reshape(len(records), 3, count).transpose(0, 2, 1)
    colors = np.concatenate([dc, rest], axis=1) if rgb else np.ones((len(records), 1), np.float32)
    with torch.cuda.device(device):
        arguments = dict(means=torch.as_tensor(data['xyz'], dtype=torch.float32, device=device),
            quats=torch.as_tensor(data['rotation'], dtype=torch.float32, device=device),
            scales=torch.as_tensor(data['scale'], dtype=torch.float32, device=device),
            opacities=torch.as_tensor(data['opacity'], dtype=torch.float32, device=device),
            colors=torch.as_tensor(colors, dtype=torch.float32, device=device),
            viewmats=torch.as_tensor(world_views, dtype=torch.float32, device=device),
            Ks=torch.as_tensor(intrinsics[indices], device=device), width=512, height=512,
            near_plane=.02*scale, far_plane=40*scale, packed=False, render_mode='RGB')
        if rgb:
            arguments.update(sh_degree=int(np.sqrt(count+1))-1, backgrounds=torch.ones((len(indices), 3), device=device))
        assert all(value.device == device for value in arguments.values() if isinstance(value, torch.Tensor))
        torch.cuda.empty_cache()
        rasterization(**arguments)
        torch.cuda.current_stream(device).synchronize()
        torch.cuda.empty_cache()
        image, alpha, _ = rasterization(**arguments)
        assert torch.isfinite(alpha).all() and torch.isfinite(image).all()
        return (image.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy() if rgb else None), alpha[..., 0].cpu().numpy()
