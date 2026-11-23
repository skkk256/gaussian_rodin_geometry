# Copyright (c) Meta Platforms, Inc. and affiliates.
import importlib.util
import os
from pathlib import Path
import sys
import torch
from typing import Optional, Dict, Any
import warnings
from torchvision.transforms import Normalize
import torch.nn.functional as F
from loguru import logger


def _load_local_dino_backbone(repo_or_dir: str, dino_model: str, backbone_kwargs: Dict[str, Any]) -> torch.nn.Module:
    """
    repo_or_dir: str, local DINOv2 source directory containing hubconf.py
    dino_model: str, hubconf callable name
    backbone_kwargs: dict, keyword arguments for the hubconf callable
    return: torch.nn.Module
    """
    repo_path = Path(repo_or_dir)
    hubconf_path = repo_path / "hubconf.py"
    assert repo_path.is_dir()
    assert hubconf_path.is_file()

    sys.path.insert(0, str(repo_path))
    try:
        spec = importlib.util.spec_from_file_location("_local_dinov2_hubconf", hubconf_path)
        assert spec is not None
        assert spec.loader is not None
        hubconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hubconf)
        assert hasattr(hubconf, dino_model)
        model_fn = getattr(hubconf, dino_model)
        return model_fn(pretrained=False, **backbone_kwargs)
    finally:
        sys.path.pop(0)


class Dino(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 224,
        repo_or_dir: str = str(Path(__file__).resolve().parents[5] / "dinov2_source"),
        dino_model: str = "dinov2_vitb14",
        source: str = "local",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        # for backward compatible
        prenorm_features: bool = False,
        freeze_backbone: bool = True,
        prune_network: bool = False,  # False for backward compatible
        checkpoint_path: str = os.environ.get("DINOV2_CHECKPOINT", str(Path(__file__).resolve().parents[6] / "ckpt/dinov2_vitl14_reg4_pretrain.pth")),
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            logger.info(f"Loading DINO model: {dino_model} from {repo_or_dir} (source: {source})")
            if backbone_kwargs:
                logger.info(f"DINO backbone kwargs: {backbone_kwargs}")
            
            assert source == "local"
            checkpoint_path = Path(checkpoint_path)
            assert checkpoint_path.is_file()
            self.backbone = _load_local_dino_backbone(repo_or_dir, dino_model, backbone_kwargs)
            
            # B. 手动加载本地 .pth 权重
            logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
            # map_location='cpu' 确保在任何设备上都能先读入，之后由 PyTorch 自动移到 GPU
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理可能的 keys 不匹配 (有些 checkpoint 会包裹在 "model" 或 "teacher" key 中)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            # 加载参数
            msg = self.backbone.load_state_dict(state_dict, strict=True)
            logger.info(f"Weights loaded with message: {msg}")


            # Log model properties after loading
            logger.info(f"Loaded DINO model - type: {type(self.backbone)}, "
                        f"embed_dim: {self.backbone.embed_dim}, "
                        f"patch_size: {getattr(self.backbone.patch_embed, 'patch_size', 'N/A')}")


        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.embed_dim
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.register_buffer('mean', torch.as_tensor([[0.485, 0.456, 0.406]]).view(-1, 1, 1), persistent=False)
        self.register_buffer('std', torch.as_tensor([[0.229, 0.224, 0.225]]).view(-1, 1, 1), persistent=False)

        # freeze
        if freeze_backbone:
            self.requires_grad_(False)
            self.eval()
        elif not prune_network:
            logger.warning(
                "Unfreeze encoder w/o prune parameter may lead to error in ddp/fp16 training"
            )

        if prune_network:
            self._prune_network()

    def _preprocess_input(self, x):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = _resized_images.sub_(self.mean).div_(self.std)

        return _resized_images

    def _forward_intermediate_layers(
        self, input_img, intermediate_layers, cls_token=True
    ):
        return self.backbone.get_intermediate_layers(
            input_img,
            intermediate_layers,
            return_class_token=cls_token,
        )

    def _forward_last_layer(self, input_img):
        output = self.backbone.forward_features(input_img)
        if self.prenorm_features:
            features = output["x_prenorm"]
            tokens = F.layer_norm(features, features.shape[-1:])
        else:
            tokens = torch.cat(
                [
                    output["x_norm_clstoken"].unsqueeze(1),
                    output["x_norm_patchtokens"],
                ],
                dim=1,
            )
        return tokens

    def forward(self, x, **kwargs):
        _resized_images = self._preprocess_input(x)
        tokens = self._forward_last_layer(_resized_images)
        return tokens.to(x.dtype)

    def _prune_network(self):
        """
        Ran this script:
        out = model(input)
        loss = out.sum()
        loss.backward()

        for name, p in dino_model.named_parameters():
            if p.grad is None:
                print(name)
        model.zero_grad()
        """
        self.backbone.mask_token = None
        if self.prenorm_features:
            self.backbone.norm = torch.nn.Identity()


class DinoForMasks(torch.nn.Module):
    def __init__(
        self,
        backbone: Dino,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

    def forward(self, image, mask):
        return self.backbone.forward(mask)
