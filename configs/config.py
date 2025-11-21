from __future__ import  annotations
from typing import Literal
from dataclasses import dataclass, field


from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, \
    FACEMESH_NOSE, FACEMESH_IRISES, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW

FACE_PARTS = [
    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW,
    FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, FACEMESH_IRISES,
    FACEMESH_NOSE,
    FACEMESH_LIPS,
]

KP_IDS_GLABELLA = [8, 9]
KP_IDS_FOREHEAD = [104, 69, 108, 151, 337, 299, 333,
                   67, 109, 10, 338, 297]
KP_IDS_NASOLABIAL = [212, 216, 206, 203, 432, 436, 426, 423,
                     57, 186, 287, 410]
KP_IDS_EYE_OUTER = [226,  35,  31, 228, 229, 230, 231, 232,
                    452, 451, 450, 449, 448, 261, 446, 265,
                    247,   30,  29,  27,  28, 56, 190,
                    414, 286, 258, 257, 259, 260, 467]

KP_IDS_GLASSES = [111, 117, 118, 119, 120, 121,
                  350, 349, 348, 347, 346, 340]

KP_IDS_IRISES = [469, 470, 471, 472, 474, 475, 476, 477]

EXPRESSION_KP_IDS = list(set(
        [item[0] for item in set().union(*FACE_PARTS)] +
        [item[1] for item in set().union(*FACE_PARTS)] +
        KP_IDS_FOREHEAD +
        KP_IDS_GLABELLA +
        KP_IDS_NASOLABIAL +
        KP_IDS_EYE_OUTER
))

ModelTypes = Literal['convmod', 'vitae', 'smooth_vitae', 'styleunet', 'styleunet_expr', 'none', 'identity', 'invresnet']

vit_params = dict(
    vitti=dict(depth=12, dim=192, mlp_dim=758, heads=3),
    vitsx=dict(depth=12, dim=256, mlp_dim=1024, heads=6),
    vits=dict(depth=12, dim=384, mlp_dim=1536, heads=6),
    vitb=dict(depth=12, dim=768, mlp_dim=3072, heads=12),
)


@dataclass
class ModelConfig:
    dino_arch: str = 'vitb14'
    with_dino: bool = True
    with_images: bool = True
    dim_expr: int = 64
    blocks_dec: int = 2
    levels_dec: int = 4
    backbone: str = 'vits'
    emb_dropout: float = 0.1

    image_size: int = 336
    num_patches: int = 24

    @property
    def vit_params(self) -> dict:
        return vit_params[self.backbone]

    @property
    def enc_emb_dim(self) -> int:
        return self.vit_params['dim']

    @property
    def channels_id(self) -> int:
        return self.vit_params['dim']

    @property
    def channels_dec(self) -> int:
        return self.vit_params['dim']

    @property
    def feature_map_size(self) -> int:
        return self.num_patches * 2**self.levels_dec
        # return 4 * 2**self.levels_dec

    color_sh_degree: int = 0
    polar: bool = False

    opt_head_pos: bool = False
    global_head_pos: bool = True

    base_radius: float = 0.20
    base_pos_x: float = 0
    base_pos_y: float = 0
    base_pos_z: float = -0.10

    k_theta: float = 0.5
    k_phi: float = 1.0
    azimuth_range: float = 180.0
    opacity: float = 0.5

    upsample: Literal['deconv', 'bilinear', 'pixel_shuffle'] = 'pixel_shuffle'


@dataclass
class RegularizationWeights:
    xyz: float = 0.1
    opac: float = 1.0
    scale: float = 0.1
    rot: float = 5.0
    shs: float = 0.0


@dataclass
class Config:
    # Model config
    net: ModelConfig = field(default_factory=ModelConfig)

    vfhq_root: str = "../../data/datasets/VFHQ"
    vfhq_data_folder: str = "processed_vfhq"

    output_dir: str = '../../data/results/results'
    checkpoint_dir: str = '../../data/results/checkpoints'

    dataset: str = "vfhq"

    validate: bool = False
    is_main_process: bool = True
    render_backend: Literal['inria', 'pytorch'] = 'inria'
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    fov: float = 30.0
    render_size: int | None = None

    cubench: bool = False
    cpu: bool = False
    show: bool = True
    wait: bool = False
    debug: bool = False
    vis_f: float = 1.0

    write_epoch_vis: bool = True
    log_epoch_vis: bool = True
    write_batch_vis: bool = True
    log_batch_vis: bool = True

    vis_freq: int = 50
    print_freq: int = 50
    save_freq: int = 1
    eval_freq: int = 5

    seed: int | None = None

    sessionname: str = "debug"
    resume: str | None = None

    # Training
    lr: float = 1e-4
    batchsize: int = 8
    n_images_per_clip: int = 3
    clip_grad_value: float = 5
    clip_grad_norm: float = -1

    init: bool = False
    startup: bool = False

    # Recon
    with_l1: bool = True
    with_ssim: bool = True
    with_sil: bool = False
    with_lpips: bool = True
    with_arcface: bool = True

    with_l1_face: bool = True
    with_ssim_face: bool = False
    with_lpips_face: bool = False

    # Feature map
    with_reg: bool = False
    with_consist: bool = False
    with_av_consist: bool = False
    with_sym: bool = False
    with_lapl: bool = False
    with_area: bool = False
    with_normal: bool = False
    with_chamfer: bool = False

    # Embedding
    with_trp: bool = False
    with_repel: bool = False
    with_center: bool = False

    with_repel_exp: bool = False
    with_center_exp: bool = False

    with_cam_pose: bool = False

    with_expr_reg: bool = False
    with_ca_param: bool = True

    w_l1: float = 0.5
    w_l1_face: float = 100. * 0.1
    w_lpips: float = 1.0
    w_lpips_face: float = 50.0
    w_ssim: float = 1.0
    w_ssim_face: float = 20.0
    w_coeff: float = 1.0
    w_reg: float = 0.1
    # w_lapl: float = 1000.0
    w_lapl: float = 1.0
    # w_consist: float = 1000.0
    w_consist: float = 10.0
    w_sil: float = 5.0
    w_normal: float = 10.0
    w_area: float = 0.001
    w_chamfer: float = 1000.0
    w_id: float = 10000.0
    w_repel: float = 1.0
    # w_sym: float = 20.0
    w_sym: float = 2.0
    w_repel_exp: float = 1.0
    w_arcface: float = 1.0

    reg: RegularizationWeights = field(default_factory=RegularizationWeights)

    # GANs
    with_gan: bool = False
    with_gan_view: bool = False
    update_D_freq: int = 2
    update_D_freq_view: int = 1
    w_gan: float = 0.1
    w_gan_view: float = 0.1
    lr_gan: float = 0.1
    lr_gan_view: float = 0.2

    pred_expr: bool = False
    pred_expr_ratio: float = 0.8
    finetune_dino: bool = False
    detach: bool = True
    grow: bool = False

    # Training data
    st: int = 0
    nd: int = 35000
    samples_per_epoch: int | None = 10000
    frames_per_clip: int = 16
    max_clip_length: int = 1000
    workers: int = 6
    remove_val_clips: bool = True
    mask_inputs: bool = False
    shuffle: bool = True
    augment: bool = False
    corrupt: bool = False
    cloth: bool = False
    latent: bool = False
    chamfer_filter: bool = True  # chamfer loss without occluded landmarks
    min_azimuth_std: float = 0
    min_azimuth_range: float = 0
    lpips_net: Literal['vgg', 'squeeze'] = 'vgg'

    arcface_model = "./assets/model_ir_se50.pth"

    # Validation
    st_val: int = 35000
    nd_val: int = 35666
    workers_val: int = 2
    batchsize_val: int = 8


default_configs = {
    "base": (
        "Base config",
        Config()
    ),
    "debug": (
        "Debug config",
        Config(
            sessionname='debug',
            st=0,
            nd=200,
            st_val=0,
            nd_val=10,
            batchsize=32,
            samples_per_epoch=1000,
            remove_val_clips=False,
            workers=0,
            workers_val=0,
            wait=True,
            vis_freq=1,
            print_freq=1,
        )
    ),
    "local": (
        "Default for training identity/shape",
        Config(
            # sessionname='debug',
            batchsize=4,
            samples_per_epoch=2500,
            eval_freq=10,
            remove_val_clips=True,
            st_val=35000,
            nd_val=36666,
            lpips_net='vgg',
            with_arcface=True
            # output_dir='../data/results/results',
            # checkpoint_dir='../data/results/checkpoints'
        )
    ),
    "server": (
        "Config for training base model on 4 x A6000 node",
        Config(
            vfhq_root="../../data/datasets/VFHQ",
            output_dir="../../data/results",
            checkpoint_dir="../../data/results/checkpoints",
            st=0,
            nd=35000,
            st_val=35000,
            nd_val=35666,
            remove_val_clips=True,
            samples_per_epoch=50000,
            show=False,
            eval_freq=1,
            save_freq=1,
            print_freq=100,
            vis_freq=100,
            batchsize=16,
        )
    ),
}