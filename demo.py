from dataclasses import dataclass
import cv2
import torch
import numpy as np

from models.live_avatar import LiveAvatar
from configs.config import ModelConfig, Config
from tracking.stream_face_tracker import StreamFaceTracker
from utils.util import transform_camera, transform_camera2
from visualization.vis import show_image


@dataclass
class Model(ModelConfig):
    image_size: int = 336
    levels_dec: int = 4
    num_patches: int = 24


@dataclass
class AvatarConfig(Config):
    net: Model
    render_size: int = 512
    device: str = "cuda"
    target_image: str = "./assets/demo_faces/mona-lisa.png"
    exp_image: str = "./assets/demo_faces/gothic_woman.png"
    checkpoint: str | None = "vitb_vits_p24_l4_pxsh_180_sh0/epoch_00724"
    pred_expr: bool = True


def rotation_matrix_y(theta: float, device="cpu"):
    """Homogeneous 4x4 rotation matrix around the world y-axis."""
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor([
        [ c, 0,  s, 0],
        [ 0, 1,  0, 0],
        [-s, 0,  c, 0],
        [ 0, 0,  0, 1],
    ], dtype=torch.float32, device=device)
    return R


def main(cfg):

    avatar = LiveAvatar(
        checkpoint=cfg.checkpoint,
        pred_expr=cfg.pred_expr,
        cfg=cfg,
    )

    target_image = cv2.cvtColor(cv2.imread(cfg.target_image), cv2.COLOR_BGR2RGB)
    exp_image = cv2.cvtColor(cv2.imread(cfg.exp_image), cv2.COLOR_BGR2RGB)
    show_image("Target image", target_image)
    show_image("Expression source", exp_image)

    # create avatar
    avatar.set_identity(target_image)
    avatar.set_expression(exp_image)
    avatar.update()

    # render same view point as expression image
    tracker = StreamFaceTracker(image_size=512, asset_dir="./assets")
    tracking_results = tracker.track(exp_image, 0, crop_input=False, show=False)

    camera = tracking_results['camera']

    for i in range(0, 3600, 2):
        theta = np.radians(i)
        M = rotation_matrix_y(theta, device=camera.device)
        new_camera = transform_camera2(camera, M)
        render = avatar.render(new_camera)
        show_image("Avatar", render.copy(), wait=5)


if __name__ == '__main__':
    import tyro
    cfg = tyro.cli(AvatarConfig)
    main(cfg)