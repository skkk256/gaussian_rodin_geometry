import torch
import nvdiffrast.torch as dr
from .reference import FIT
from .rotation_math import LIMIT

def unpack(raw):
    """raw trainable float32 [7] -> bounded scale scalar, offset [3], rotation matrix [3,3], rotation vector [3]."""
    scale = 1+.2*torch.tanh(raw[0])
    offset = .15*torch.tanh(raw[1:4])
    norm = torch.linalg.vector_norm(raw[4:]).clamp_min(1e-8)
    r = LIMIT*torch.tanh(norm)/norm*raw[4:]
    x, y, z = r.unbind()
    zero = torch.zeros((), device=raw.device)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
    theta = torch.linalg.vector_norm(r)
    R = torch.eye(3, device=raw.device)+torch.sinc(theta/torch.pi)*skew+.5*torch.sinc(theta/(2*torch.pi))**2*(skew @ skew)
    return scale, offset, R, r

def alpha_render(renderer, vertices, raw):
    """bbox vertices float32 [M,3], parameters float32 [7] -> differentiable alpha float32 [8,512,512]."""
    assert vertices.device == raw.device == renderer.device
    assert renderer.vertices.device == renderer.faces.device == renderer.views.device == renderer.intrinsics.device == renderer.device
    scale, offset, R, _ = unpack(raw)
    xyz = scale*(vertices @ R.T)+offset
    views, k = renderer.views[FIT], renderer.intrinsics[FIT]
    camera = torch.einsum('vij,nj->vni', views[:, :3, :3], xyz)+views[:, None, :3, 3]
    z = camera[..., 2]
    clip = torch.stack([2*k[:, 0, 0, None]*camera[..., 0]/512,
        -2*k[:, 1, 1, None]*camera[..., 1]/512, (40.02/39.98)*z-1.6/39.98, z], dim=-1).contiguous()
    raster, _ = dr.rasterize(renderer.context, clip, renderer.faces, (512, 512))
    occupied = (raster[..., 3:] > 0).to(torch.float32)
    return dr.antialias(occupied, raster, clip, renderer.faces)[..., 0].flip(1)
