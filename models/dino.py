import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import kornia as K


def get_pretrained_dinov2(arch) -> torch.nn.Module:
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    assert arch in backbone_archs.values()
    # backbone_arch = backbone_archs[arch]
    backbone_name = f"dinov2_{arch}"

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    for p in model.parameters():
        p.requires_grad = False
    return model


class DINOFusionSimple(nn.Module):
    def __init__(self, in_dim, output_dim: int = None, with_images=False):
        super().__init__()
        self.project = nn.Identity()
        self.with_images = with_images
        if output_dim is None:
            self.output_conv = nn.Identity()
        else:
            self.output_conv = nn.Conv2d(in_dim * 4 + (128 if with_images else 0), output_dim, kernel_size=1, stride=1, padding=0)

    def forward(
            self,
            dino_features: list[torch.Tensor],
            img_features: torch.Tensor | None = None,
            output_size: None | int | list[int] = None
    ):

        dino_patch_h, dino_patch_w = [int(dino_features[0].shape[1] ** 0.5)] * 2

        def reshape_feature(feature):
            return feature.permute(0, 2, 1).reshape(
                (feature.shape[0], feature.shape[-1], dino_patch_h, dino_patch_w)
            )

        # features = self.project(features)

        features = torch.cat([reshape_feature(feature) for feature in dino_features], dim=1)

        if output_size is not None:
            if isinstance(output_size, int):
                output_size = [output_size, output_size]
            if output_size[0] != dino_patch_h or output_size[1] != dino_patch_w:
                features = torchvision.transforms.functional.resize(features, output_size, antialias=True)

        if self.with_images and img_features is not None:
            img_features = F.interpolate(
                img_features, output_size, mode='bilinear', align_corners=False)
            self.img_features = img_features

            features = torch.cat([features, img_features], dim=1)

        return self.output_conv(features)