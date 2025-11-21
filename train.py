from __future__ import annotations

# Get rid of warnings from packages
import warnings
from models.recog.facerec import FaceRec
from utils.util import stack_cameras

warnings.filterwarnings("ignore")

import itertools

# Get rid of stacktrace output after pressing Ctrl+C
import sys
import signal
def handler(signum, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, handler)

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import pandas as pd
import time
import numpy as np
import datetime
import torch
import torch.utils.data as td
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import kornia as K
import albumentations as alb
import random

from torchmetrics.image import PeakSignalNoiseRatio
from accelerate import Accelerator
from pytorch_msssim import SSIM

from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import lpips
import tyro
from torch.utils.tensorboard import SummaryWriter

from training import base_training
from rendering.renderer import GaussRenderer
from models.neural_head_net import NeuralHeadNet
from training.loss_utils import laplacian_loss_conv, symmetry_loss, regularization_loss
from utils.nn import to_numpy
from utils import log
from utils.util import crop_and_resize_from_keypoints
from visualization.vis import add_landmarks_to_images, show_image, write_image, make_grid
from visualization.batch_vis import BatchVis, visualize_gaussian_maps


def replace_background(images, masks, new_color=(1, 1, 1)):
    m = masks.float()
    background = (1 - m) * torch.ones_like(m)
    images[:, 0] = background + masks.float() * images[:, 0]
    images[:, 1] = background + masks.float() * images[:, 1]
    images[:, 2] = background + masks.float() * images[:, 2]
    return images


def get_view_from_same_clip(img_id, clip_ids):
    id_current_clip = clip_ids[img_id]
    candidates = set(np.where(clip_ids == id_current_clip)[0])
    if len(candidates) > 1:
        candidates.remove(img_id)
    rnd_cand_id = np.random.randint(low=0, high=len(candidates))
    return list(candidates)[rnd_cand_id]


def shuffle_by_clipid(image_ids, clip_ids):
    result = np.zeros_like(clip_ids)
    image_ids = np.array(image_ids)
    for cl in np.unique(clip_ids):
        indices = np.where(clip_ids == cl)[0]
        img_ids = image_ids[indices]
        shuffled_ids = np.random.permutation(img_ids)
        result[indices] = shuffled_ids
    return result


def create_image_weights(target_images, masks):
    B, C, H, W = target_images.shape
    num_fg_pixels_per_image = masks.reshape(B, -1).sum(dim=1)
    weights = torch.nan_to_num(1.0 / num_fg_pixels_per_image, nan=0, posinf=0)
    weights[num_fg_pixels_per_image < 0.25 * H * W] = 0
    return weights


def head_l1_loss(preds, targets):
    return F.l1_loss(preds, targets) * 20


def seed_worker(worker_id):
    worker_seed = 0
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class RandomZoomOut(alb.ImageOnlyTransform):
    def __init__(self, min_scale=1.0, max_scale=1.5, p=1.0, pad_value=0):
        super().__init__(p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.pad_value = pad_value

    def apply(self, img, **params):
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Random scale (>=1 for zoom out)
        scale = random.uniform(self.min_scale, self.max_scale)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Determine padding
        pad_top = max((new_h - h) // 2, 0)
        pad_bottom = new_h - h - pad_top
        pad_left = max((new_w - w) // 2, 0)
        pad_right = new_w - w - pad_left

        # Pad image
        padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=self.pad_value)

        # Random crop of original size from padded
        # y1 = random.randint(0, padded.shape[0] - h)
        # x1 = random.randint(0, padded.shape[1] - w)
        # cropped = padded[y1:y1 + h, x1:x1 + w]

        return padded


def interleave(A: torch.Tensor | list, B: torch.Tensor | list) -> torch.Tensor | list:

    if isinstance(A, list):
        return [x for pair in zip(A, B) for x in pair]

    # Check same shape
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    # Stack -> [*shape, 2]
    stacked = torch.stack((A, B), dim=1)   # adds a new axis for interleave
    # Reshape so interleave happens along the first axis
    return stacked.reshape(-1, *A.shape[1:])


class AvatarTraining(base_training.Training):

    def __init__(
            self,
            dataloaders: dict[str, td.DataLoader],
            net: NeuralHeadNet,
            gaussian_renderer: GaussRenderer,
            lr: float = 1e-3,
            **kwargs
    ):
        super().__init__(dataloaders, net, lr, **kwargs)

        self.gaussian_render = gaussian_renderer
        self.gpc_params = net.gpc_params
        self.n_fixed = min(10, self.cfg.batchsize)

        if self.cfg.with_ssim:
            self._ssim = SSIM(data_range=1.0, size_average=True, channel=3)

        self._lpips = None
        if self.cfg.with_lpips:
            self._lpips = lpips.LPIPS(net=cfg.lpips_net).to(self.device)

        self.face_rec = None
        if self.cfg.with_arcface:
            self.face_rec = FaceRec(cfg.arcface_model, self.device)

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction=None, dim=(1,2,3)).to(self.device)

        # Logging and visualization is only done on the main process
        if self.accelerator.is_local_main_process:
            # Set-up tensorboard logging
            strdate = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            log_dir = os.path.join(self.cfg.output_dir, 'tensorboard', self.cfg.sessionname, strdate)
            self.writer = SummaryWriter(str(log_dir), flush_secs=10)

            # Set-up visualizer
            self._vis = BatchVis(
                net=net,
                gaussian_renderer=self.gaussian_render,
            )

        #
        # Create optimizers
        #

        opt_param_list = [
            self.net.dino_fusion.parameters(),
            self.net.enc_img.parameters(),
            self.net.transformer.parameters(),
            self.net.P_id.parameters(),
        ]

        self.optimizer = torch.optim.Adam(list(itertools.chain(*opt_param_list)), lr=lr)
        self.optimizer_pos = torch.optim.Adam([self.net.pos, self.net.scale], lr=0.001)

        if self.cfg.pred_expr:
            cross_attn_gammas = [layer[0].gamma for layer in self.net.transformer.transformer.layers]
            # style_proj_layers = [layer[0].style_proj for layer in self.net.transformer.transformer.layers]
            # style_params = [layer.parameters() for layer in style_proj_layers]
            # style_params = list(itertools.chain(*style_params))
            opt_params_expr = (
                    list(self.net.enc_img.parameters()) +
                    list(self.net.dino_fusion.parameters()) +
                    list(self.net.dino_fusion_expr.parameters()) +
                    list(self.net.enc_expr.parameters()) +
                    list(self.net.transformer.parameters()) +
                    list(self.net.P_id.parameters()) +
                    cross_attn_gammas
            )
            self.optimizer = torch.optim.Adam(opt_params_expr, lr=lr)
            # if self.cfg.pred_expr:
            #     self.net.requires_grad_(False)
            #     for param in style_params:
            #         param.requires_grad = True

        if self.cfg.grow:
            # layers = [layer for layer in self.net.P_id.layers]
            trainable_dec_params =  list(self.net.P_id.layers[-1].parameters()) + list(self.net.P_id.out.parameters())
            self.optimizer = torch.optim.Adam(trainable_dec_params, lr=lr)

        # return deterministic copy of dataloader with new batch size
        def get_fixed_batchloader(dataloader: td.DataLoader, num_images: int) -> td.DataLoader:
            if cfg.n_images_per_clip > 1:
                return td.DataLoader(dataloader.dataset, batch_size=num_images, num_workers=0, shuffle=True,
                                     generator=torch.Generator().manual_seed(0), worker_init_fn=seed_worker)
            else:
                return td.DataLoader(dataloader.dataset, batch_size=num_images, num_workers=0, shuffle=True)

        # save some samples to visualize the training progress
        fixed_batch_loader_train = get_fixed_batchloader(self.dataloaders['train'], self.n_fixed)
        fixed_batch_loader_val = get_fixed_batchloader(self.dataloaders['val'], self.n_fixed)

        assert isinstance(self.net, NeuralHeadNet)  # get type info for IDE

        #
        # Prepare for multi-gpu training
        #

        modules_to_prepare = [
            self.net,
            self.optimizer,
            self.optimizer_pos,
            self.dataloaders['train'],
            self.dataloaders['val'],
            fixed_batch_loader_train,
            fixed_batch_loader_val,
        ]

        prepared_modules = accelerator.prepare(*modules_to_prepare)

        [
            self.net,
            self.optimizer,
            self.optimizer_pos,
            self.dataloaders['train'],
            self.dataloaders['val'],
            fixed_batch_loader_train,
            fixed_batch_loader_val,
        ] = prepared_modules

        self.fixed_batch_train = next(iter(fixed_batch_loader_train))
        self.fixed_batch_val = next(iter(fixed_batch_loader_val))

        # from torchsummary import summary
        # summary(self.net, input_size=(3, 140, 140))
        # print("summary")

    def _get_metrics_msg(self, means):
        msg_metrics = "loss={loss:.4f}: "

        if self.cfg.with_l1: msg_metrics += "l1={loss_l1:.3f} "
        if self.cfg.with_l1_face: msg_metrics += "l1f={loss_l1_face:.3f} "
        if self.cfg.with_ssim: msg_metrics += "ssim={loss_ssim:.3f} "
        if self.cfg.with_ssim_face: msg_metrics += "ssimf={loss_ssim_face:.3f} "
        if self.cfg.with_lpips: msg_metrics += "lpips={loss_lpips:.3f} "
        if self.cfg.with_lpips_face: msg_metrics += "lpipsf={loss_lpips_face:.3f} "
        if self.cfg.with_reg: msg_metrics += "reg={loss_reg:.3f} "
        if self.cfg.with_lapl: msg_metrics += "lapl={loss_lapl:.3f} "
        if self.cfg.with_normal: msg_metrics += "n={loss_normal:.3f} "
        if self.cfg.with_area: msg_metrics += "a={loss_area:.3f} "
        if self.cfg.with_chamfer: msg_metrics += "chf={loss_chamfer:.3f} "
        if self.cfg.with_sym: msg_metrics += "sym={loss_sym:.3f} "
        if self.cfg.with_av_consist: msg_metrics += "av={loss_av_consist:.3f} "
        if self.cfg.with_consist: msg_metrics += "view={loss_consist:.3f} "
        if self.cfg.with_arcface: msg_metrics += "arc={loss_arcface:.3f} "
        if self.cfg.with_ca_param: msg_metrics += "ca={loss_ca_param:.3f} "
        if self.cfg.with_expr_reg: msg_metrics += "eflip={loss_expr_reg:.3f} "
        if self.cfg.with_sil: msg_metrics += "sil={loss_sil:.3f} "
        if self.cfg.with_cam_pose: msg_metrics += "R|T={loss_cam_rot:.3f}|{loss_cam_pos:.3f} "
        if self.cfg.with_gan: msg_metrics += "l_D={loss_D:.3f} l_G={loss_G:.3f} "
        if self.cfg.with_gan_view: msg_metrics += "l_Dn={loss_Dn:.3f} l_Gn={loss_Gn:.3f} "
        if self.cfg.with_center_exp: msg_metrics += "e_ctr={loss_center_exp:.3f} "
        if self.cfg.with_repel_exp: msg_metrics += "e_rpl={loss_repel_exp:.3f} "

        msg_metrics += "psnr={psnr:.2f}  "

        msg_metrics = msg_metrics.format(
            loss=means.get('loss', -1),
            loss_l1=means.get('loss_l1', -1),
            loss_l1_face=means.get('loss_l1_face', -1),
            loss_lpips_face=means.get('loss_lpips_face', -1),
            loss_ssim=means.get('loss_ssim', -1),
            loss_ssim_face=means.get('loss_ssim_face', -1),
            loss_lpips=means.get('loss_lpips', -1),
            loss_sil=means.get('loss_sil', -1),
            loss_reg=means.get('loss_reg', -1),
            loss_lapl=means.get('loss_lapl', -1),
            loss_normal=means.get('loss_normal', -1),
            loss_area=means.get('loss_area', -1),
            loss_chamfer=means.get('loss_chamfer', -1),
            loss_sym=means.get('loss_sym', -1),
            loss_consist=means.get('loss_consist', -1),
            loss_av_consist=means.get('loss_av_consist', -1),
            loss_repel_exp=means.get('loss_repel_exp', -1),
            loss_center_exp=means.get('loss_center_exp', -1),
            loss_arcface=means.get('loss_arcface', -1),
            loss_expr_reg=means.get('loss_expr_reg', -1),
            loss_ca_param=means.get('loss_ca_param', -1),
            psnr=means.get('psnr', -1),
        )
        return msg_metrics

    def _print_iter_stats(self, stats):
        if not self.is_main_process:
            return
        means = pd.DataFrame(stats)
        means = means.mean().to_dict()
        current = stats[-1]
        msg_head = "[{ep}][{i}/{iters_per_epoch}] ".format(
            ep=current['epoch'] + 1,
            i=current['iter'] + 1,
            iters_per_epoch=self.iters_per_epoch,
        )
        self._stats_message_head_length = len(msg_head)
        msg_metrics = self._get_metrics_msg(means)
        msg_tail = "{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})".format(
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        )
        log.info(msg_head + msg_metrics + msg_tail)

    def _print_epoch_summary(self, epoch_stats, epoch_starttime, is_eval=False):
        if not self.is_main_process:
            return

        means = pd.DataFrame(epoch_stats)
        means = means.mean().to_dict()

        duration = int(time.time() - epoch_starttime)
        log.info("{}".format('-' * 100))

        msg_head = " " * self._stats_message_head_length
        msg_metrics = self._get_metrics_msg(means)
        msg_tail = " \tT: {time_epoch}".format(
            time_epoch=str(datetime.timedelta(seconds=duration))
        )
        log.info(msg_head + msg_metrics + msg_tail)

    def _on_epoch_end(self, is_eval):
        super()._on_epoch_end(is_eval)

        # run on fixed batch to track progress
        data = self.fixed_batch_val if is_eval else self.fixed_batch_train

        log_dict, figures = self._run_batch(data, is_eval=True, visualize=True, tracking=False)

        # only the main process will create figures
        if figures is not None:
            for fig_name, fig_img in figures.items():
                split = 'val' if is_eval else 'train'
                out_dir = os.path.join(self.cfg.output_dir, 'vis', self.session_name, split)
                os.makedirs(out_dir, exist_ok=True)
                if self.cfg.write_epoch_vis:
                    img_filepath = os.path.join(out_dir, f"{fig_name}_epoch_{self.epoch + 1:05d}.jpg")
                    write_image(img_filepath, fig_img)
                if self.cfg.log_epoch_vis:
                    self.writer.add_image(f"{split}/{fig_name}", fig_img, global_step=self.total_iter,
                                          walltime=self.total_training_time(),  dataformats='HWC')

    def _compute_losses(
            self,
            target_images: torch.Tensor,
            results: dict,
            pixel_weights: torch.Tensor,
            keypoints: torch.Tensor,
            pred_expr: bool
    ):
        losses = {}

        pred_images = results['pred_images']
        _, _, H, W = pred_images.shape

        # If render size is different from image size, rescale target images
        # to compute losses on render size
        if target_images.shape != pred_images.shape:
            target_images = K.geometry.resize(target_images, (H, W))
            pixel_weights = K.geometry.resize(pixel_weights, (H, W))

        kp2d = keypoints[..., :2] * torch.tensor([W, H]).to(self.device)
        targets_crop = crop_and_resize_from_keypoints(target_images, kp2d, output_size=(H, W))
        preds_crop = crop_and_resize_from_keypoints(pred_images, kp2d, output_size=(H, W))

        lowres_size = (224, 224)
        targets_crop_lr = crop_and_resize_from_keypoints(target_images, kp2d, output_size=lowres_size)
        preds_crop_lr = crop_and_resize_from_keypoints(pred_images, kp2d, output_size=lowres_size)

        #######################
        # Reconstruction
        #######################

        if self.cfg.with_l1:
            losses['loss_l1'] = head_l1_loss(pred_images, target_images) * self.cfg.w_l1

        if self.cfg.with_lpips:
            preds_resized = torch.nn.functional.interpolate(pred_images, size=224, mode="bilinear")
            targets_resized = torch.nn.functional.interpolate(target_images, size=224, mode="bilinear")
            losses['loss_lpips'] = self._lpips(preds_resized, targets_resized, normalize=True).mean() * self.cfg.w_lpips
            # losses['loss_lpips'] = self._lpips(pred_images, target_images, normalize=True).mean() * self.cfg.w_lpips
            # losses['loss_lpips'] = self._lpips(preds_crop_lr, targets_crop_lr, normalize=True).mean() * self.cfg.w_lpips

        if self.cfg.with_ssim:
            losses['loss_ssim'] = (1.0 - self._ssim(pred_images, target_images)) * self.cfg.w_ssim

        # Focus on facial features relevant for expressions
        if pred_expr:
            # target_images_face = target_images * pixel_weights
            # pred_images_face = pred_images * pixel_weights
            # losses['loss_l1_face'] = F.l1_loss(pred_images_face, target_images_face) * self.cfg.w_l1_face

            # targets_crop = self.net.zoom_in_outputs(target_images_face)
            # preds_crop = self.net.zoom_in_outputs(pred_images_face)
            # show_image("targets crop", make_grid(targets_crop), wait=1)
            # show_image("preds crop", make_grid(preds_crop), wait=0)
            losses['loss_l1_face'] = F.l1_loss(preds_crop, targets_crop) * self.cfg.w_l1_face

        #######################
        # Regularizers
        #######################

        if self.cfg.with_reg:
            feature_maps = results['feature_maps']
            losses['loss_reg'] = regularization_loss(feature_maps, self.gpc_params, self.cfg.reg) * self.cfg.w_reg

        #######################
        # Arcface loss
        #######################

        if self.cfg.with_arcface:
            emb_target = self.face_rec.embed(targets_crop_lr)
            emb_pred = self.face_rec.embed(preds_crop_lr)
            losses['loss_arcface'] = (1.0 - torch.nn.functional.cosine_similarity(emb_pred, emb_target).mean()) * self.cfg.w_arcface

        if self.cfg.with_sym:
            losses['loss_sym'] = symmetry_loss(results['feature_maps'], self.gpc_params) * self.cfg.w_sym

        if self.cfg.with_lapl:
            losses['loss_lapl'] = laplacian_loss_conv(results['feature_maps']) * 10.0

        return losses

    def _run_batch(self, batch, is_eval=False, visualize: bool = False, tracking=True):

        is_train = not is_eval
        self.net.train(is_train)

        _pred_expr = self.cfg.pred_expr
        if _pred_expr and is_train:
            skip_interval = 5
            _pred_expr = (self.iter_in_epoch % skip_interval) != (skip_interval-1)

        log_dict = {}

        loss_dict = dict(
            loss=torch.zeros(1, requires_grad=True, device=self.device),
            loss_l1 = torch.zeros(1, requires_grad=True, device=self.device),
            loss_l1_face = torch.zeros(1, requires_grad=True, device=self.device),
            loss_ssim = torch.zeros(1, requires_grad=True, device=self.device),
            loss_ssim_face = torch.zeros(1, requires_grad=True, device=self.device),
            loss_lpips_face = torch.zeros(1, requires_grad=True, device=self.device),
            loss_lpips = torch.zeros(1, requires_grad=True, device=self.device),
            loss_consist = torch.zeros(1, requires_grad=True, device=self.device),
            loss_chamfer = torch.zeros(1, requires_grad=True, device=self.device),
            loss_sym = torch.zeros(1, requires_grad=True, device=self.device),
            loss_arcface = torch.zeros(1, requires_grad=True, device=self.device),
            loss_reg = torch.zeros(1, requires_grad=True, device=self.device),
            loss_ca_param = torch.zeros(1, requires_grad=True, device=self.device),
            loss_lapl=torch.zeros(1, requires_grad=True, device=self.device),
        )

        ##############################################
        #
        # Forward model
        #
        ##############################################

        input_images: torch.Tensor = batch[0]['input']
        target_cameras = []
        expr_images = []
        keypoints_aligned = []

        # if is_train:
        target_views_ids = range(len(batch))
        # else:
        #     target_views_ids = [1]

        camera_distance = 1.0 / np.tan(np.radians(30.0)) * 1.
        cam_pos = np.array([[0., 0., camera_distance]]).repeat(4,0)
        look_at = np.array([[0., 0., 0.0]]).repeat(4, 0)
        R, T = look_at_view_transform(eye=cam_pos, at=look_at)
        _cam = FoVPerspectiveCameras(R=R, T=T, fov=30.0, device=self.device)

        for view_id in target_views_ids:
            target_cameras.append(FoVPerspectiveCameras(
                R=batch[view_id]['R'],
                T=batch[view_id]['T'],
                fov=self.cfg.fov,
                device=self.device
            ))
            # target_cameras.append(_cam)
            expr_images.append(batch[view_id]['input'])
            # keypoints_aligned.append(batch[view_id]['keypoints_aligned'])
            keypoints_aligned.append(batch[view_id]['keypoints'])

        results = self.net(
            input_images,
            x_exp_list=expr_images,
            cameras_list=target_cameras,
            keypoints_list=keypoints_aligned,
            gaussian_renderer=self.gaussian_render,
            is_train=is_train,
            pred_expr=_pred_expr,
        )

        ##############################################
        #
        # Compute losses
        #
        ##############################################

        num_results = len(results)
        if is_eval:
            num_results = 1

        # accumulate losses from individual target views
        for target_view_id in range(len(results)):
            if is_eval and target_view_id == 0:
                continue
            target_images: torch.Tensor = batch[target_view_id]['target']
            face_weights: torch.Tensor = batch[target_view_id]['face_weights']
            losses = self._compute_losses(
                target_images,
                results[target_view_id],
                face_weights,
                keypoints=batch[target_view_id]['keypoints'],
                pred_expr=_pred_expr
            )
            for k in losses.keys():
                loss_dict[k] = loss_dict[k] + losses[k] / num_results

        # sum total loss
        for k in loss_dict.keys():
            loss_dict['loss'] = loss_dict['loss'] + loss_dict[k]

        ##############################################
        #
        # Logging and metrics
        #
        ##############################################

        if self.is_main_process:

            for k in loss_dict:
                log_dict[k] = loss_dict[k].item()

            log_dict['psnr'] = self.psnr_metric(
                results[-1]['pred_images'],
                batch[-1]['target']
            ).mean().item()

            if tracking:
                prefix = "train" if is_train else "val"
                self.writer.add_scalar(f"{prefix}/loss", log_dict['loss'], self.total_iter, walltime=self.total_training_time())
                self.writer.add_scalar(f"{prefix}/l1", log_dict['loss_l1'], self.total_iter, walltime=self.total_training_time())
                self.writer.add_scalar(f"{prefix}/ssim", log_dict['loss_ssim'], self.total_iter, walltime=self.total_training_time())
                self.writer.add_scalar(f"{prefix}/lpips", log_dict['loss_lpips'], self.total_iter, walltime=self.total_training_time())
                self.writer.add_scalar(f"{prefix}/psnr", log_dict.get('psnr', 0), self.total_iter, walltime=self.total_training_time())
                # self.writer.add_scalars(prefix, log_dict, self.total_iter, walltime=self.total_training_time())

        ##############################################
        #
        # Backward model
        #
        ##############################################

        if is_train:

            accelerator.backward(loss_dict['loss'])

            if self.cfg.clip_grad_value > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_value_(self.net.parameters(), self.cfg.clip_grad_value)

            if self.cfg.clip_grad_norm > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.net.parameters(), self.cfg.clip_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimizer_pos.step()
            self.optimizer_pos.zero_grad()

        ##############################################
        #
        # Visualization
        #
        ##############################################

        figures = None

        if self.is_main_process:

            if visualize:

                self.net.eval()

                disp_batch = self._vis.show_batch(
                    batch,
                    input_images,
                    cameras=target_cameras[0],
                    results=results,
                    pred_expr=_pred_expr
                )

                target_images = batch[-1]['target']
                pred_images = results[-1]['pred_images']
                embeddings = results[-1]['embeddings']

                disp_recons = self._vis.show_reconstructions(
                    batch,
                    input_images,
                    target_images,
                    pred_images,
                    max_images=8,
                    status_bar_text=self._get_metrics_msg(log_dict)
                )

                disp_gaussians = visualize_gaussian_maps(
                    results[0]['pointclouds'][:1], results[0]['feature_maps'][:1], fov=self.cfg.fov
                )

                figures = dict(
                    batch=disp_batch,
                    recons=disp_recons,
                    gaussian_deltas=disp_gaussians,
                    # embeddings=self._vis.show_embeddings(input_images, embeddings, source_image_ids, max_images=8),
                    # random_id=self._vis.visualize_random_id(embeddings),
                    interp_id=self._vis.visualize_interpolations(
                        embeddings, embeddings, idx1=0, idx2=1, camera1=target_cameras[0][0],
                        camera2=target_cameras[0][1])
                )
                if _pred_expr:
                    figures['random_expr'] = self._vis.visualize_random_expr(embeddings)
                    figures['expr_matrix'] = self._vis.create_expression_matrix(
                        embeddings, target_cameras[-1], input_images=batch[-1]['input']
                    )

                if self.cfg.write_batch_vis:
                    out_dir = os.path.join(self.cfg.output_dir, 'vis', self.session_name, "current")
                    os.makedirs(out_dir, exist_ok=True)
                    for fig_name, fig_img in figures.items():
                        img_filepath = os.path.join(out_dir, f"{fig_name}.jpg")
                        write_image(img_filepath, fig_img)

                # show image on screen
                if self.cfg.show:
                    for fig_name, fig_img in figures.items():
                        show_image(fig_name, fig_img)
                    cv2.waitKey(0 if self.cfg.wait else 1)

        return log_dict, figures


if __name__ == '__main__':
    from albumentations.pytorch import transforms as alb_torch
    from datasets.vfhq import VFHQ
    from configs.config import default_configs
    from accelerate.utils import set_seed
    from accelerate import DistributedDataParallelKwargs

    cfg = tyro.extras.overridable_config_cli(default_configs)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = cfg.cubench

    if cfg.seed is not None:
        log.info(f"Setting seed={cfg.seed}")
        set_seed(cfg.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=cfg.cpu, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device

    cfg.is_main_process = accelerator.is_local_main_process

    if cfg.debug:
        cfg.workers = 0
        cfg.workers_val = 0
        cfg.print_freq = 1
        cfg.vis_freq = 1
        cfg.wait = 0

    if cfg.init:
        cfg.with_l1 = False
        cfg.with_chamfer = False

    if cfg.startup:
        cfg.with_l1 = True
        cfg.with_chamfer = True
        cfg.with_sil = True

    # if cfg.pred_expr:
        # cfg.with_l1_face = True
        # cfg.with_center_exp = False

    if not cfg.latent:
        cfg.with_trp = False
        cfg.with_center = False
        cfg.with_repel = False

    if cfg.render_size is None:
        cfg.render_size = cfg.net.image_size

    if cfg.is_main_process:
        log.info(cfg)
        log.info(f"Benchmark: {torch.backends.cudnn.benchmark}")

    transform = alb.Compose([
        # alb.Resize(height=cfg.net.render_size, width=cfg.net.render_size),
        alb.Resize(height=cfg.render_size, width=cfg.render_size),
        alb_torch.ToTensorV2()
    ])

    # pixelwise_transform = alb.Compose([
    #     alb.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
    #     alb.HueSaturationValue(p=1.0),
    #     alb.RGBShift(p=1),
    #     alb.RandomGamma(p=1),
    # ])

    corruption_transform = None

    dataset_val = VFHQ(
        os.path.join(cfg.vfhq_root, 'test'),
        data_folder=cfg.vfhq_data_folder,
        transform=transform,
        train=False,
        mask_inputs=cfg.mask_inputs,
        cloth=cfg.cloth,
        return_face_weights=cfg.with_l1_face,
        background_color=cfg.background_color,
        n_images_per_clip=2,
        reenact=True,
    )
    dataset_train = VFHQ(
        os.path.join(cfg.vfhq_root, 'train'),
        max_clip_length=cfg.max_clip_length,
        num_frames=cfg.frames_per_clip,
        transform=transform,
        # pixelwise_transform=pixelwise_transform,
        corruption_transform=corruption_transform,
        st=cfg.st,
        nd=cfg.nd,
        # blacklist_clips=dataset_val.get_clip_names() if cfg.remove_val_clips else None,
        # blacklist_videos=dataset_val.get_video_names() if cfg.remove_val_clips else None,
        train=True,
        data_folder=cfg.vfhq_data_folder,
        mask_inputs=cfg.mask_inputs,
        cloth=cfg.cloth,
        return_face_weights=cfg.with_l1_face,
        background_color=cfg.background_color,
        filter=dict(
            min_azimuth_std=cfg.min_azimuth_std,
            min_azimuth_range=cfg.min_azimuth_range,
        ),
        n_images_per_clip=cfg.n_images_per_clip
    )

    dataloaders = {
        'train': td.DataLoader(dataset_train, cfg.batchsize, num_workers=cfg.workers, pin_memory=True,
                               sampler=torch.utils.data.RandomSampler(dataset_train,  num_samples=cfg.samples_per_epoch)),
        'val': td.DataLoader(dataset_val, batch_size=cfg.batchsize_val, num_workers=cfg.workers_val,
                             shuffle=True, generator=torch.Generator().manual_seed(0),
                             worker_init_fn=seed_worker),
    }

    if cfg.is_main_process:
        log.info("")
        log.info(dataset_train)
        log.info(dataset_val)
        if 'test' in dataloaders:
            log.info(dataloaders['test'].dataset)
        log.info("")

    net = NeuralHeadNet(
        params=cfg.net,
        device=device,
        train=True,
        train_cfg=cfg,
    )

    if accelerator.num_processes > 1:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    render_size = cfg.render_size if cfg.render_size is not None else cfg.net.image_size
    gaussian_renderer = GaussRenderer(render_size, render_size, background_color=cfg.background_color,
                                      render_backend=cfg.render_backend)

    trainer = AvatarTraining(
        net=net,
        gaussian_renderer=gaussian_renderer,
        dataloaders=dataloaders,
        lr=cfg.lr,
        session_name=cfg.sessionname,
        resume=cfg.resume,
        snapshot_dir=cfg.checkpoint_dir,
        vis_freq=cfg.vis_freq,
        print_freq=cfg.print_freq,
        save_freq=cfg.save_freq,
        eval_freq=cfg.eval_freq,
        config=cfg,
        accelerator=accelerator
    )

    if cfg.validate:
        trainer.validate()
    else:
        trainer.train()
