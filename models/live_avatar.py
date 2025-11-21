from __future__ import annotations
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras

from models.gaussian_pointclouds import GaussianPointclouds
from rendering.renderer import GaussRenderer
from visualization.vis import to_image
from models.neural_head_net import NeuralHeadNet, load_model
# from tto import TTO


def blend_embeddings(emb_list: list[dict], weights: list[float]) -> dict:
    if len(emb_list) == 1:
        return emb_list[0]

    if weights is None:
        weights = [1.0] * len(emb_list)

    weights = np.array(weights)/sum(weights)

    blended_ft_id = [emb['ft_id']*w for emb, w in zip(emb_list, weights)]
    blended_ft_id = torch.stack(blended_ft_id).sum(dim=0)
    embeddings = {
        'ft_id': blended_ft_id,
        'ft_expr': emb_list[-1]['ft_expr']
    }
    return embeddings


class LiveAvatar():
    def __init__(self, checkpoint, pred_expr, cfg):
        net = NeuralHeadNet(
            params=cfg.net,
            # pred_expr=pred_expr,
            device=cfg.device,
            train_cfg=cfg,
            tto=False,
        )
        self.net = load_model(net, checkpoint, cfg.checkpoint_dir).to(cfg.device).eval()
        self.gaussian_renderer = GaussRenderer(cfg.render_size, cfg.render_size, background_color=cfg.background_color)

        import torchvision.transforms as transforms
        self.transform = transforms.Compose([
            transforms.Resize((net.params.image_size, net.params.image_size)),
            transforms.ToTensor(),
        ])
        self._tick = None
        self._tock = time.time()
        self.embeddings = []
        self.blend_weights = None

    def reset(self) -> None:
        self.embeddings = []

    def fps(self) -> float:
        if self._tock is None or self._tick is None:
            return 0
        return 1.0 / (self._tock - self._tick)

    def _to_input(self, image: np.ndarray):
        return self.transform(Image.fromarray(image)).to(self.net.device).unsqueeze(0)

    def set_identity(self, image: np.ndarray) -> None:
        self.identity_image = self._to_input(image)
        with torch.no_grad():
            self.embeddings = [self.net.encode_identity(self.identity_image, return_dino=False)]
            self.blend_weights = [1.0]

    def add_identity(self, image: np.ndarray) -> None:
        self.identity_image = self._to_input(image)
        with torch.no_grad():
            self.embeddings.append(self.net.encode_identity(self.identity_image, return_dino=False))
            self.blend_weights.append(0.0)

    def remove_identity(self, idx: int = 0) -> None:
        if len(self.embeddings) > idx:
            del self.embeddings[idx]
            del self.blend_weights[idx]

    def set_expression(self, image: np.ndarray, keypoints: np.ndarray=None) -> None:
        if self.embeddings is None:
            self.embeddings = self.set_identity(image)
        self.input_image = self._to_input(image)
        if keypoints is not None:
            keypoints = torch.tensor(keypoints).to(self.net.device)
            if len(keypoints.shape) == 2:
                keypoints = keypoints.unsqueeze(0)
        self.embeddings[-1]['ft_expr'] = self.net.encode_expressions(self.input_image, keypoints=keypoints)['ft_expr']

    def set_identity_and_expression(self, image: np.ndarray, keypoints=None) -> None:
        self.identity_image = self._to_input(image)
        self.input_image = self._to_input(image)
        if keypoints is not None:
            keypoints = torch.tensor(keypoints).to(self.net.device)
            if len(keypoints.shape) == 2:
                keypoints = keypoints.unsqueeze(0)
        with torch.no_grad():
            self.embeddings = [self.net.encode(self.input_image, keypoints=keypoints, pred_expr=True)]
            self.blend_weights = [1.0]

    def update(self):
        embeddings = blend_embeddings(self.embeddings, self.blend_weights)

        with torch.no_grad():
            self.pcs = self.net.decode(embeddings)['pointclouds']
            # crop pointcloud
            B, C, H, W = self.pcs._features.shape
            self.pcs = GaussianPointclouds(
                features=self.pcs._features[:, :, :, int(W*0.10):-int(W*0.10)],
                pos=self.pcs._pos,
                scales=self.pcs._scales,
                params=self.pcs._params
            )

        self._tick = self._tock
        self._tock = time.time()

    # def create(self, image: np.ndarray) -> LiveAvatar:
    #     with torch.no_grad():
    #         input_image = self._to_input(image)
    #         results = self.net(input_image, pred_expr=True)
    #     self.embeddings = [results['embeddings']]
    #     self.pcs = results['pointclouds']
    #     self._tick = self._tock
    #     self._tock = time.time()
    #     self.identity_image = image
    #     self.input_image = image
    #     return self

    def render(self, camera: FoVPerspectiveCameras) -> np.ndarray:
        with torch.no_grad():
            pred_images = self.gaussian_renderer.render_images(self.pcs, camera.to(self.net.device))
        return to_image(pred_images)