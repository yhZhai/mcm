# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""
import logging
import os
from typing import List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import Normalize, RandomCrop

from .diffaug import DiffAugment
from .shared import ResidualBlock

logger = logging.getLogger(__name__)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name="weight", n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(
        self,
        num_features: int,
        affine: bool = True,
        virtual_bs: int = 8,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead(nn.Module):
    def __init__(
        self,
        channels: int,
        c_dim: Optional[int] = None,
        combine_clip: bool = False,
        cmap_dim: int = 64,
    ):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.combine_clip = combine_clip
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9)),
        )

        if not self.combine_clip:
            if self.c_dim is not None and self.c_dim > 0:
                self.cmapper = nn.Linear(self.c_dim, cmap_dim)
                self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
            else:
                self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)
        else:
            self.patch_cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

            self.main_cls_token = nn.Sequential(
                make_block(channels, kernel_size=1),
                ResidualBlock(make_block(channels, kernel_size=1)),
            )
            self.cmapper = nn.Linear(self.c_dim, cmap_dim)
            self.cls_cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: [b, c, n], c: [b, c_dim]
        if not self.combine_clip:
            h = self.main(x)
            out = self.cls(h)

            if self.c_dim is not None and self.c_dim > 0:
                cmap = self.cmapper(c).unsqueeze(-1)
                out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        else:
            patch_token = x[:, :, :-1]
            patch_token = self.main(patch_token)
            patch_out = self.patch_cls(patch_token)
            patch_out = patch_out.mean(2, keepdim=True)

            cls_token = x[:, :, -1:]
            cls_token = self.main_cls_token(cls_token)
            cls_token = self.cls_cls(cls_token)
            cmap = self.cmapper(c).unsqueeze(-1)
            cls_out = (cls_token * cmap).sum(1, keepdim=True) * (
                1 / np.sqrt(self.cmap_dim)
            )
            out = torch.cat([patch_out, cls_out], dim=2)

        return out


class ProjectedDiscriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        embed_dim: int = 384,
        c_dim: Optional[int] = None,
        combine_clip: bool = False,
        diffaug: bool = True,
        p_crop: float = 0.5,
        resize: int = 224,
        hooks: List[int] = [2, 5, 8, 11],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.c_dim = c_dim
        self.combine_clip = combine_clip
        self.diffaug = diffaug
        self.p_crop = p_crop
        self.hooks = hooks
        self.resize = resize

        heads = []
        for i in hooks:
            heads += ([str(i), DiscHead(self.embed_dim, c_dim, combine_clip)],)
        self.heads = nn.ModuleDict(heads)

    def forward(
        self,
        feature_list: List,
        c: Optional[torch.Tensor] = None,
        return_key_list: List[str] = ["logits"],
    ) -> torch.Tensor:
        input_list = []
        logit_list = []
        logits = []
        for idx, layer_idx in enumerate(self.heads.keys()):
            feature = feature_list[idx]
            feature = rearrange(feature, "b c n -> b n c")  # [b, c, num_patch]
            b = feature.shape[0]
            head_out = self.heads[layer_idx](feature, c).view(b, -1)
            input_list.append(feature)
            logit_list.append(head_out)
        logits = torch.cat(logit_list, dim=1)

        return_dict = {}
        if "logits" in return_key_list:
            return_dict["logits"] = logits
        if "input_list" in return_key_list:
            return_dict["input_list"] = input_list
        if "logit_list" in return_key_list:
            return_dict["logit_list"] = logit_list

        return return_dict


def preprocess_dino_input(
    x,
    diffaug: bool = True,
    resize: int = 224,
    p_crop: float = 0.5,
    norm_fn: Optional[Normalize] = None,
):
    # x in [-1, 1], shape: b, c, h, w
    # c shape: b, c_dim
    # Apply augmentation (x in [-1, 1]).
    if diffaug:
        x = DiffAugment(x, policy="color,translation,cutout")

    # Transform to [0, 1].
    x = x.add(1).div(2)

    # Take crops with probablity p_crop if the image is larger.
    if x.size(-1) > resize and np.random.random() < p_crop:
        x = RandomCrop(resize)(x)

    # Forward pass through DINO ViT.
    if resize > 0:
        x = F.interpolate(x, resize, mode="area")

    if norm_fn is not None:
        x = norm_fn(x)
    return x


def concat_dino_cls_token(feature_list: List, do_concat: bool):
    feature_list = list(feature_list)
    if not do_concat:
        return feature_list

    for i in range(len(feature_list)):
        if not isinstance(feature_list[i], torch.Tensor):
            cls_token = feature_list[i][1]
            cls_token = cls_token.unsqueeze(1)
            feature_list[i] = torch.cat([feature_list[i][0], cls_token], dim=1)
    return feature_list


def get_dino_features(
    sample,
    normalize_fn,
    dino_model,
    dino_hooks: List[int],
    return_cls_token: bool,
    preprocess_sample: bool = True,
):
    if preprocess_sample:
        sample = preprocess_dino_input(sample, norm_fn=normalize_fn)
    dino_features = dino_model.get_intermediate_layers(
        sample, n=dino_hooks, return_class_token=return_cls_token
    )
    dino_features = concat_dino_cls_token(dino_features, return_cls_token)
    return dino_features, sample


if __name__ == "__main__":
    disc = DiscHead(384, 768, combine_clip=True)
    feature = torch.randn(2, 384, 17)
    c_feature = torch.randn(2, 768)

    out = disc(feature, c_feature)
    print(out.shape)
