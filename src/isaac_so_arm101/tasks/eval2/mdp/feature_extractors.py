# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DINOv2 model zoo entries for ``isaaclab.envs.mdp.image_features``.

Pass ``DINOV2_MODEL_ZOO`` as the ``model_zoo_cfg`` param and set
``model_name`` to one of its keys (e.g. ``"dinov2_vits14"``) to swap the
default ResNet18 feature extractor for a frozen DINOv2 ViT.

All callables are defined at module level so Hydra can serialize the cfg
(it converts callables into ``module:attr`` strings, which would fail for
closures).

Output is the ``[CLS]`` class token per image — shape ``(N, embed_dim)``:
    dinov2_vits14  →  (N, 384)
    dinov2_vitb14  →  (N, 768)
"""

from __future__ import annotations

import torch


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dinov2_vits14() -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.cuda()


def _load_dinov2_vitb14() -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.cuda()


def _inference_dinov2(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    # images: (N, H, W, C) uint8 from TiledCamera
    device = next(model.parameters()).device
    x = images.to(device)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        features = model(x)
    return features


DINOV2_MODEL_ZOO = {
    "dinov2_vits14": {
        "model": _load_dinov2_vits14,
        "inference": _inference_dinov2,
    },
    "dinov2_vitb14": {
        "model": _load_dinov2_vitb14,
        "inference": _inference_dinov2,
    },
}
