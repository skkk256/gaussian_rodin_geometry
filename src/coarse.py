from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix

os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform


class CoarseInference:
    def __init__(self, config_file: Path, device: str):
        assert torch.cuda.is_available(), "CoarseInference requires CUDA"
        indexed_device = torch.device(device)
        assert indexed_device.type == "cuda" and indexed_device.index is not None
        assert indexed_device.index < torch.cuda.device_count()
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = str(config_file.parent)
        config.device = str(indexed_device)
        gpu_name = torch.cuda.get_device_name(indexed_device)
        if any(name in gpu_name for name in ('A100', 'H100', 'H200')):
            os.environ['ATTN_BACKEND'] = 'flash_attn'
            os.environ['SPARSE_ATTN_BACKEND'] = 'flash_attn'
        with torch.cuda.device(indexed_device):
            self.pipeline = instantiate(config)
        for model in self.pipeline.models.values():
            if model is not None:
                model.requires_grad_(False)
        for embedder in self.pipeline.condition_embedders.values():
            if embedder is not None:
                embedder.requires_grad_(False)
        self.device = indexed_device
        self.warmed = False

    def generate_coarse(
        self, image: np.ndarray, mask: np.ndarray, pointmap: torch.Tensor, seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        image: np.ndarray [H, W, 3] uint8 RGB
        mask: np.ndarray [H, W] bool
        pointmap: torch.Tensor [H, W, 3] float32 PyTorch3D camera coordinates
        return: coords np.ndarray [N, 3] int, transform np.ndarray [4, 4] float
        """
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert mask.dtype == np.bool_ and mask.shape == image.shape[:2]
        assert pointmap.shape == (*mask.shape, 3)
        rgba = np.concatenate((image, (mask[..., None] * 255).astype(np.uint8)), axis=2)
        arguments = dict(
            image=rgba, mask=None, seed=seed, stage1_only=True,
            with_mesh_postprocess=False, with_texture_baking=False,
            with_layout_postprocess=True, use_vertex_color=True,
            stage1_inference_steps=10, pointmap=pointmap,
        )
        with torch.cuda.device(self.device):
            if not self.warmed:
                torch.cuda.empty_cache()
                warmup = self.pipeline.run(**arguments)
                del warmup
                torch.cuda.current_stream(self.device).synchronize()
                torch.cuda.empty_cache()
                self.warmed = True
            output = self.pipeline.run(**arguments)
        transform = compose_transform(
            scale=output["scale"],
            rotation=quaternion_to_matrix(output["rotation"]),
            translation=output["translation"],
        ).get_matrix().reshape(4, 4).T.cpu().numpy()
        coords = output["coords"][:, 1:4].cpu().numpy()
        assert coords.shape[0] > 0 and coords.shape[1] == 3
        assert (coords >= 0).all() and (coords < 64).all()
        return coords, transform
