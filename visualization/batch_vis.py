from __future__ import annotations
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from scipy.spatial.distance import pdist, squareform

from models.gaussian_pointclouds import GaussianPointclouds
from models.neural_head_net import NeuralHeadNet, normalize_identity
from rendering.renderer import GaussRenderer
from models.gaussian_pointclouds import PointcloudConverter
import torch.nn.functional as F

from pytorch3d.renderer import (
    # look_at_view_transform,
    FoVPerspectiveCameras,
    # PointLights,
    # Materials,
    # RasterizationSettings,
    # MeshRenderer,
    # MeshRasterizer,
    # TexturesVertex,
    # HardGouraudShader,
    # BlendParams
)

from utils.util import crop_and_resize_from_keypoints
from utils.nn import to_numpy
from utils.util import create_camera, backproject_landmarks, stack_cameras
from visualization.pcavis import PCAVis
from visualization.vis import (
    draw_channels,
    draw_colors,
    make_grid,
    draw_channel,
    add_label_to_images,
    to_cmap_images,
    add_error_to_images, show_image,
    draw_status_bar, add_landmarks_to_images, color_map

)
import kornia as K
import torchmetrics


def str_frame_info(clip_id, clip_name, frame_id):
    return f"{clip_id}:{clip_name}/{frame_id}"


def visualize_gaussian_maps(pc: GaussianPointclouds, feature_maps: torch.Tensor, fov, output_size=None):
    assert len(pc) == 1

    def draw_feature_channels(name):
        range = pc._params.scale_factors[name]
        return [draw_channel(c, vmin=-range, vmax=range, cmap='viridis') for c in feature_maps[0][pc._params.channels[name]]]

    img_xyz = draw_channels(pc.xyz[0])

    distance = 1.0 / np.tan(np.radians(fov))
    cam_pos = torch.tensor([0., 0., distance], device=pc.device)
    rgb_colors = pc[0].get_colors(camera_position=cam_pos, return_map=True)
    img_colors = draw_colors(rgb_colors)

    img_opacity = draw_channel(torch.sigmoid(pc.opacity[0]), vmin=0, vmax=1.0)

    imgs_coord = draw_feature_channels('coords')
    imgs_scale = draw_feature_channels('scaling')
    imgs_rot = draw_feature_channels('rotation')
    # imgs_shs = draw_feature_channels('shs')

    items = [
        *imgs_coord,
        img_xyz,
        img_opacity,
        *imgs_scale,
        *imgs_rot,
        # *imgs_shs,
        img_colors
    ]

    grid = make_grid(items, nrows=1, padsize=0)
    if output_size is not None:
        grid = cv2.resize(grid, dsize=(len(items) * output_size, output_size), interpolation=cv2.INTER_NEAREST)
    return grid


def visualize_embeddings_pca(
        features: torch.tensor | np.ndarray,
        output_size: tuple[int, int] | None = None
) -> list[np.ndarray]:
    pcavis = PCAVis(segment=False)
    pcavis.fit(features)
    features_rgb = pcavis.transform(features)
    if output_size is not None:
        features_rgb = [cv2.resize(img, dsize=(output_size[0], output_size[1]), interpolation=cv2.INTER_NEAREST)
                        for img in features_rgb]
    return features_rgb


def visualize_feature_vectors(
        features: torch.tensor | np.ndarray,
        output_size: tuple[int, int],
        vmin=-1.0,
        vmax=1.0
) -> list[np.ndarray]:
    width, height = output_size
    def convert(ft):
        return cv2.resize(
            color_map(to_numpy(ft.reshape(1, -1)), vmin=vmin, vmax=vmax),
            dsize=(width, height),
            interpolation=cv2.INTER_NEAREST
        )
    return [convert(features[i]) for i in range(features.shape[0])]


def create_interpolated_vectors(v1, v2, nsteps, mode='real2real'):
    assert nsteps >= 2
    if mode == 'real2real':
        st = v1
        nd = v2
    elif mode == 'rea2random':
        st = v1
        nd = torch.randn_like(v2)
    elif mode == 'random2random':
        st = torch.randn_like(v1)
        nd = torch.randn_like(v2)
    else:
        raise ValueError(f"Unknow mode {mode}")
    return torch.stack([st + (nd - st)/(nsteps-1) * i for i in range(nsteps)])


def plot_tsne(features):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    features = features[::2]
    N = len(features)

    if N < 16:
        return

    randoms = normalize_identity(torch.randn_like(features))
    data = torch.vstack([features.reshape(N, -1), randoms.reshape(N, -1)])

    tsne = TSNE()
    X = tsne.fit_transform(to_numpy(data))

    fig,ax = plt.subplots(2)
    ax[0].scatter(X[:N, 1], X[:N, 0])
    ax[0].scatter(X[N:, 1], X[N:, 0])

    ax[1].plot(to_numpy(features[0].ravel()))
    ax[1].plot(to_numpy(randoms[0].ravel()))
    # plt.show()

    def mplfig_to_numpy(fig):
        import io
        with io.BytesIO() as io_buf:
            fig.savefig(io_buf, format='raw')
            io_buf.seek(0)
            data = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        return im

    show_image("tsne", mplfig_to_numpy(fig))


class BatchVis():
    def __init__(
            self,
            net: NeuralHeadNet,
            gaussian_renderer: GaussRenderer,
    ):
        super().__init__()
        self.gaussian_render = gaussian_renderer
        self.net = net
        device = net.device

        fov = net.train_cfg.fov
        self.cam_front_left = create_camera(45, 0, fov=fov).to(device)
        self.cam_front_mid = create_camera(0, 0, fov=fov).to(device)
        self.cam_front_right = create_camera(-45, 0, fov=fov).to(device)
        cam_angles_color = [90, 45, 0, -45, -90]
        cam_angles_shape = [135, 90, 0, -90, -135]
        self.fixed_cams_color = [create_camera(angle, 0, fov=fov).to(device) for angle in cam_angles_color]
        self.fixed_cams_shape = [create_camera(angle, 0, fov=fov).to(device) for angle in cam_angles_shape]

        from pytorch_msssim import SSIM
        self._ssim = SSIM(data_range=1.0, size_average=False, channel=3)
        self.psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0, reduction=None, dim=(1,2,3)).to(device)
        self.lpips_model = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True, reduction='none').to(device)

    def show_batch(
            self,
            batch: dict,
            input_images: torch.Tensor,
            cameras: FoVPerspectiveCameras,
            results: list[dict],
            results_noexp: list[dict] | None = None,
            novel_views: torch.Tensor | None = None,
            pred_expr: bool = False,
            max_images: int = 4,
    ):
        B, C, H, W = results[0]['pred_images'].shape
        nimgs = min(B, max_images)
        nimgs_per_row = 1
        padval = 255

        inputs = input_images[:nimgs].clone()

        # resize input images to match targets so they can be placed in same image grid
        sources = K.geometry.resize(inputs, (H, W))

        panels = []

        # Input images
        frame_info = [str_frame_info(batch[0]['clip_id'][i], batch[0]['clip_name'][i], batch[0]['fid'][i])
                      for i in range(nimgs)]
        sources = add_label_to_images(sources, frame_info, size=0.4)
        panels.append(make_grid(sources, ncols=nimgs_per_row, padval=padval))

        # Target images
        for target_view_id in range(len(batch)):
            targets = batch[target_view_id]['target'][:nimgs].clone()
            target_cameras = FoVPerspectiveCameras(
                R=batch[target_view_id]['R'],
                T=batch[target_view_id]['T'],
                fov=self.net.train_cfg.fov,
                device=self.net.device
            )
            frame_info = [str_frame_info(batch[target_view_id]['clip_id'][i],
                                         batch[target_view_id]['clip_name'][i],
                                         batch[target_view_id]['fid'][i]) for i in range(nimgs)]
            targets = add_label_to_images(targets, frame_info, size=0.4)
            # panels.append(make_grid(targets, ncols=nimgs_per_row, padval=padval))

            # kp3d = to_numpy(batch[target_view_id]['keypoints_aligned'][..., :3])
            # keypoints_proj = torch.stack(
            #     [backproject_landmarks(kp3d[i], target_cameras[i], W, H) for i in range(nimgs)]
            # )
            # targets = add_landmarks_to_images(targets, keypoints_proj)
            panels.append(make_grid(targets, ncols=nimgs_per_row, padval=padval))

        # Target silhouettes
        # panels.append(
        #     make_grid(to_cmap_images(target_silhouettes[:nimgs].float()), ncols=nimgs_per_row, padval=padval)
        # )

        # Predictions
        for res in results:
            panels.append(make_grid(res['pred_images'][:nimgs], ncols=nimgs_per_row, padval=padval))
            # kp2d = to_numpy(batch[1]['keypoints'][..., :2]) * np.array([H, W])
            # panels.append(make_grid(add_landmarks_to_images(predicted_images[:nimgs], kp2d), ncols=nimgs_per_row, padval=padval))

        if results_noexp is not None:
            for res in results_noexp:
                panels.append(make_grid(res['pred_images'][:nimgs], ncols=nimgs_per_row, padval=padval))

        # black spacer
        panels.append(make_grid(torch.zeros_like(results[0]['pred_images'][:nimgs]), ncols=nimgs_per_row, padval=padval))

        self.net.eval()

        if novel_views is None:
            dists = squareform(pdist(to_numpy(batch[0]['azimuth'].unsqueeze(1))))
            shuffled_ids = dists.argmax(axis=1)
            # shuffled_ids = np.random.permutation(range(len(cameras)))
            shuffled_cameras = stack_cameras([cameras[int(i)] for i in shuffled_ids])

            embeddings = results[0]['embeddings']

            with torch.no_grad():
                novel_views = self.gaussian_render.render_images(
                    self.net.decode(embeddings)['pointclouds'][:nimgs],
                    shuffled_cameras
                )

            def shuffle_expressions(embs, new_ids):
                return {
                    'ft_id': embs['ft_id'],
                    'ft_expr': embs['ft_expr'][new_ids] if embs['ft_expr'] is not None else None
                }

            with torch.no_grad():
                pcs = self.net.decode(shuffle_expressions(embeddings, shuffled_ids))['pointclouds'][:nimgs]
                novel_views_with_expr = self.gaussian_render.render_images(pcs, shuffled_cameras)
            grid_novel_views = make_grid(novel_views[:nimgs], ncols=nimgs_per_row, padval=padval)
            grid_novel_views_with_expr = make_grid(novel_views_with_expr[:nimgs], ncols=nimgs_per_row, padval=padval)

            panels += [grid_novel_views]
            panels += [grid_novel_views_with_expr]

        #
        # Create frontal/side views
        #
        with torch.no_grad():
            pcs = self.net(inputs,
                           x_exp_list=[batch[0]['input'][:nimgs]],
                           keypoints_list=[batch[0]['keypoints'][:nimgs]],
                           pred_expr=pred_expr)[0]['pointclouds']

            B, C, H, W = pcs._features.shape
            cropped_pcs = GaussianPointclouds(
                features=pcs._features[:, :, :, int(W*0.10):-int(W*0.10)],
                pos=pcs._pos,
                scales=pcs._scales,
                params=pcs._params
            )
            grids_views_color = [
                make_grid(self.gaussian_render.render_images(cropped_pcs, cam),
                          ncols=nimgs_per_row, padval=padval) for cam in self.fixed_cams_color
            ]
        panels += grids_views_color

        # pcs_base = GaussianPointclouds(features=outputs['base_maps'], pos=outputs['pos'],
        #                                scales=outputs['scales'], params=self.net.gpc_params)
        # cam_backside = create_camera(180, 0, fov=30).to('cuda')
        # list_grids += [
        #     self.create_grid_with_normal_maps(pcs_base, self.fixed_cams_shape[len(self.fixed_cams_shape) // 2])
        #     # self.create_grid_with_normal_maps(pcs_base, cam_backside)
        # ]

        if self.net.train_cfg.with_l1_face:
            target_images = batch[-1]['target'][:nimgs]
            pred_images = results[-1]['pred_images'][:nimgs]
            # pixel_weights = batch[0]['face_weights'][:nimgs]
            # target_images_face = target_images * pixel_weights
            # panels += [make_grid(target_images_face, ncols=nimgs_per_row, padval=padval)]

            B, C, H, W = target_images.shape
            kp2d = batch[-1]['keypoints'][:nimgs, :, :2] * torch.tensor([W, H]).to(self.net.device)
            targets_crop = crop_and_resize_from_keypoints(target_images, kp2d, output_size=(H, W))
            preds_crop = crop_and_resize_from_keypoints(pred_images, kp2d, output_size=(H, W))
            panels += [make_grid(targets_crop, ncols=nimgs_per_row, padval=padval)]
            panels += [make_grid(preds_crop, ncols=nimgs_per_row, padval=padval)]

        grid = make_grid(panels, ncols=len(panels), padsize=0)
        new_height = 900
        new_width = int(grid.shape[1] * (new_height / grid.shape[0]))
        return cv2.resize(grid, dsize=(new_width, new_height))


    def show_reconstructions(
            self,
            batch: dict,
            input_images: torch.Tensor,
            target_images: torch.Tensor,
            predicted_images: torch.Tensor,
            max_images: int,
            predicted_images_expr: torch.Tensor = None,
            source_image_ids: list[int] = None,
            status_bar_text: str = None
    ) -> np.ndarray:
        B, C, H, W = target_images.shape
        nimgs = min(B, max_images)
        image_pixel_count = H*W

        if source_image_ids is None:
            source_image_ids = range(len(input_images))

        targets = target_images[:nimgs].clone()
        sources = K.geometry.resize(input_images[source_image_ids][:nimgs], (H, W))
        # masks = batch['mask'][:nimgs].cuda()
        # fg_pixels_per_image = masks.reshape(masks.shape[0], -1).sum(dim=1)

        if predicted_images is None:
            predicted_images = torch.zeros_like(target_images)

        preds = predicted_images[:nimgs].clone()

        l1 = torch.nn.functional.l1_loss(preds, targets, reduction='none')
        l1_per_image = l1.reshape(l1.shape[0], -1).mean(dim=1) * 20 * 2.0
        ssim_per_image = self._ssim(preds, targets) #* (image_pixel_count / fg_pixels_per_image)
        psnr_per_image = self.psnr_metric(preds, targets)
        lpips_per_image = self.lpips_model(preds, targets)

        frame_labels = [str_frame_info(batch[0]['clip_id'][i], batch[0]['clip_name'][i], batch[0]['fid'][i]) for i in source_image_ids]
        sources = add_label_to_images(sources, frame_labels, size=0.4)

        frame_labels = [str_frame_info(batch[-1]['clip_id'][i], batch[-1]['clip_name'][i], batch[-1]['fid'][i]) for i in range(nimgs)]
        targets = add_label_to_images(targets, frame_labels, size=0.4)

        preds = add_error_to_images(preds, l1_per_image, loc='tl', size=0.6, vmin=0.8, vmax=2.0)
        preds = add_error_to_images(preds, ssim_per_image, loc='tr', size=0.6, vmax=0.5)
        preds = add_label_to_images(preds, psnr_per_image, loc='tr+1', size=0.6)
        preds = add_label_to_images(preds, lpips_per_image, loc='tr+2', size=0.6)
        # preds = add_error_to_images(preds, fg_pixels_per_image, loc='bl', size=0.4)

        l1_maps = to_cmap_images(l1.mean(dim=1), cmap=plt.cm.jet, vmax=1.0)
        l1_maps = add_error_to_images(l1_maps, l1_per_image, loc='tl', size=0.5, vmax=1.0)

        # ssim_maps = to_cmap_images(ssim.mean(dim=1), cmap=plt.cm.jet, vmax=1.0)
        # ssim_maps = add_error_to_images(ssim, l1_per_image, loc='tl', size=0.5, vmax=1.0)

        panels = [
            make_grid(sources, nrows=1),
            make_grid(targets, nrows=1),
            make_grid(preds, nrows=1),
            make_grid(l1_maps, nrows=1),
        ]

        if predicted_images_expr is not None:
            preds_expr = predicted_images_expr[:nimgs].clone()
            panels.append(make_grid(preds_expr, nrows=1))

        disp_recons = make_grid(panels,  ncols=1, padsize=0)

        if H > 500:
            disp_recons = cv2.resize(disp_recons, dsize=None, fx=0.60, fy=0.60, interpolation=cv2.INTER_LANCZOS4)

        if status_bar_text is not None:
            status_bar = draw_status_bar(status_bar_text,
                                         status_bar_width=disp_recons.shape[1],
                                         status_bar_height=30,
                                         dtype=disp_recons.dtype,
                                         text_color=(255, 255, 255))
            disp_recons = np.vstack([disp_recons, status_bar])

        return disp_recons

    def show_embeddings(
            self,
            input_images: torch.Tensor,
            embeddings: dict,
            source_image_ids: list[int] = None,
            max_images: int = 12
    ) -> np.ndarray:
        B, C, H, W = input_images.shape
        nimgs = min(B, max_images)

        if source_image_ids is None:
            source_image_ids = range(len(input_images))

        sources = input_images[source_image_ids][:nimgs].clone()

        images_ft_dino = visualize_embeddings_pca(embeddings['ft_id'][source_image_ids][:nimgs], output_size=(W, H))
        # images_ft_img = visualize_embeddings_pca(embeddings['ft_img'][source_image_ids][:nimgs], output_size=(W, H))
        # images_ft_id = visualize_embeddings_pca(embeddings['ft_id'][source_image_ids][:nimgs], output_size=(W, H))
        images_ft_av = visualize_embeddings_pca(embeddings['ft_av'][source_image_ids][:nimgs], output_size=(W, H))

        panels = [
            make_grid(sources, nrows=1),
            make_grid(images_ft_dino, nrows=1),
            # make_grid(images_ft_img, nrows=1),
            # make_grid(images_ft_id, nrows=1),
            make_grid(images_ft_av, nrows=1),
        ]
        if embeddings['ft_expr'] is not None:
            panels.append(
                make_grid(visualize_feature_vectors(embeddings['ft_expr'][:nimgs], output_size=(W, 40)))
            )

        return np.vstack(panels)

    def interpolate(self, embeddings1: dict, embeddings2: dict, idx1, idx2,
                    camera1: FoVPerspectiveCameras, camera2: FoVPerspectiveCameras,
                    ninterp=9):
        assert len(embeddings1['ft_id'].shape) == 4
        assert embeddings1['ft_id'].shape == embeddings2['ft_id'].shape

        ft_id_interp = create_interpolated_vectors(
            embeddings1['ft_id'][idx1],
            embeddings2['ft_id'][idx2],
            nsteps=ninterp,
        ).to(self.net.device)
        ft_id_interp = normalize_identity(ft_id_interp)

        rot6d_interp = create_interpolated_vectors(
            matrix_to_rotation_6d(camera1.R),
            matrix_to_rotation_6d(camera2.R),
            nsteps=ninterp
        )
        R_interp = rotation_6d_to_matrix(rot6d_interp).squeeze(1)
        T_interp = create_interpolated_vectors(camera1.T, camera2.T, nsteps=ninterp).squeeze(1)

        cameras_interp = FoVPerspectiveCameras(R=R_interp, T=T_interp, fov=camera1.fov).to(self.net.device)

        ft_expr_interp = None
        if embeddings1['ft_expr'] is not None:
            ft_expr_interp = create_interpolated_vectors(
                embeddings1['ft_expr'][idx1],
                embeddings2['ft_expr'][idx2],
                nsteps=ninterp,
            ).to(self.net.device)
            ft_expr_interp = ft_expr_interp

        embeddings_interp = {
            'ft_id': ft_id_interp,
            'ft_expr': ft_expr_interp,
        }
        return embeddings_interp, cameras_interp

    def visualize_interpolations_2d(self, embeddings: dict, idx1, idx2, idx3, idx4,
                                 cameras: FoVPerspectiveCameras, ninterp=9):

        def render(embs, cams):
            return self.gaussian_render.render_images(self.net.decode(embs)['pointclouds'], cams)

        # create top row
        emb_interp_row1, cam_interp_row1 = self.interpolate(
            embeddings, embeddings, idx1, idx2, camera1=cameras[idx1], camera2=cameras[idx2], ninterp=ninterp
        )

        # create bottom row
        emb_interp_row2, cam_interp_row2 = self.interpolate(
            embeddings, embeddings, idx3, idx4, camera1=cameras[idx3], camera2=cameras[idx4], ninterp=ninterp
        )

        show_image("row1", make_grid(render(emb_interp_row1, cam_interp_row1)), wait=1, f=0.3)
        show_image("row2", make_grid(render(emb_interp_row2, cam_interp_row2)), wait=1, f=0.3)

        # interpolate between top row and bottom row
        images = []
        with torch.no_grad():
            for i in range(len(emb_interp_row1['ft_expr'])):
                emb_interp_col, cam_interp_col = self.interpolate(
                    emb_interp_row1, emb_interp_row2, i, i, camera1=cam_interp_row1[i], camera2=cam_interp_row2[i],
                    ninterp=ninterp
                )
                images.extend(render(emb_interp_col, cam_interp_col))
                # pointclouds = self.net.decode(emb_interp_col)['pointclouds']
                # images.extend(
                #     self.gaussian_render.render_images(pointclouds, cam_interp_col)
                # )

        return make_grid(images, nrows=ninterp, ncols=ninterp)


    def visualize_interpolations(self, embeddings1: dict, embeddings2: dict, idx1, idx2,
                                 camera1: FoVPerspectiveCameras, camera2: FoVPerspectiveCameras,
                                 ninterp=9):
        assert len(embeddings1['ft_id'].shape) == 4
        assert embeddings1['ft_id'].shape == embeddings2['ft_id'].shape

        ft_id_interp = create_interpolated_vectors(
            embeddings1['ft_id'][idx1],
            embeddings2['ft_id'][idx2],
            nsteps=ninterp,
        ).to(self.net.device)
        ft_id_interp = normalize_identity(ft_id_interp)

        # ft_coarse_interp = embeddings1['ft_coarse'][idx1].unsqueeze(0).repeat(ninterp, 1),
        # ft_coarse_interp = embeddings1['ft_coarse']
        # if embeddings1['ft_coarse'] is not None and embeddings2['ft_coarse'] is not None:
        #     ft_coarse_interp = create_interpolated_vectors(
        #         embeddings1['ft_coarse'][idx1],
        #         embeddings2['ft_coarse'][idx2],
        #         nsteps=ninterp,
        #     ).to(self.net.device)

        rot6d_interp = create_interpolated_vectors(
            matrix_to_rotation_6d(camera1.R),
            matrix_to_rotation_6d(camera2.R),
            nsteps=ninterp
        )
        R_interp = rotation_6d_to_matrix(rot6d_interp).squeeze(1)
        T_interp = create_interpolated_vectors(camera1.T, camera2.T, nsteps=ninterp).squeeze(1)

        cameras_interp = FoVPerspectiveCameras(R=R_interp, T=T_interp, fov=camera1.fov).to(self.net.device)

        ft_expr_interp = None
        if embeddings1['ft_expr'] is not None:
            ft_expr_interp = create_interpolated_vectors(
                embeddings1['ft_expr'][idx1],
                embeddings2['ft_expr'][idx2],
                nsteps=ninterp,
            ).to(self.net.device)
            # ft_expr_interp = F.normalize(ft_expr_interp)
            ft_expr_interp = ft_expr_interp

        # ft_expr = embeddings1['ft_expr'][idx1].unsqueeze(0).repeat(ninterp, 1) if embeddings1['ft_expr'] is not None else None

        embeddings_interp = {
            'ft_id': ft_id_interp,
            # 'ft_coarse': ft_coarse_interp,
            'ft_expr': ft_expr_interp,
        }
        with torch.no_grad():
            pointclouds = self.net.decode(embeddings_interp)['pointclouds']
            images = self.gaussian_render.render_images(pointclouds, cameras_interp)
        return make_grid(images, nrows=1)


    def visualize_random_id(self, embeddings, cameras=None, num=16):
        num = min(len(embeddings['ft_id']), num)

        # sample random points on hypersphere
        ft_id_random = normalize_identity(torch.randn_like(embeddings['ft_id']))

        # plot_tsne(embeddings['ft_id'])

        # ft_coarse_constant = embeddings['ft_coarse']
        # if embeddings['ft_coarse'] is not None:
        #     ft_coarse_constant = embeddings['ft_coarse'][0].unsqueeze(0).repeat(num, 1)

        embeddings_random = {
            'ft_id': ft_id_random[:num],
            # 'ft_coarse': ft_coarse_constant,
            'ft_expr': None,
        }
        if cameras is None:
            cameras = self.cam_front_mid
        with torch.no_grad():
            pointclouds = self.net.decode(embeddings_random)['pointclouds']
            images = self.gaussian_render.render_images(pointclouds, cameras)
        return make_grid(images, ncols=4)

    def visualize_random_expr(self, embeddings, cameras=None, num=16):
        num = min(len(embeddings['ft_expr']), num)

        # sample random points on hypersphere
        ft_expr_random = F.normalize(torch.rand_like(embeddings['ft_expr']))

        # plot_tsne(embeddings['ft_expr'])

        # ft_coarse_constant = embeddings['ft_coarse']
        # if embeddings['ft_coarse'] is not None:
        #     ft_coarse_constant = embeddings['ft_coarse'][0].unsqueeze(0).repeat(num, 1)

        embeddings_random = {
            'ft_id': embeddings['ft_id'][:num],
            # 'ft_coarse': ft_coarse_constant,
            'ft_expr': ft_expr_random[:num],
        }
        if cameras is None:
            cameras = self.cam_front_mid
        with torch.no_grad():
            pointclouds = self.net.decode(embeddings_random)['pointclouds']
            images = self.gaussian_render.render_images(pointclouds, cameras)
        disp_expr = make_grid(images, ncols=4)

        if disp_expr.shape[0] > 500:
            disp_expr = cv2.resize(disp_expr, dsize=None, fx=0.70, fy=0.70, interpolation=cv2.INTER_LANCZOS4)
        return disp_expr

    def create_expression_matrix(self, embeddings, cameras, input_images=None, source_image_ids=None, num=6, padval=255):
        num = min(len(embeddings['ft_id']), num)
        H, W = input_images.shape[2:]

        if source_image_ids is None:
            source_image_ids = range(len(embeddings['ft_id']))

        def make_row(source_id):
            source_ids = [source_id] * num

            change_expression = False

            if change_expression:
                embeddings_new = {
                    'ft_id': embeddings['ft_id'][:num],
                    'ft_expr': embeddings['ft_expr'][source_ids],
                }
                cameras_row = cameras[source_id]
            else: # change identity
                embeddings_new = {
                    'ft_id': embeddings['ft_id'][source_image_ids][source_ids],
                    'ft_expr': embeddings['ft_expr'][:num],
                }
                cameras_row = cameras

            # if embeddings['ft_coarse'] is not None:
            #     embeddings_new['ft_coarse'] = embeddings['ft_coarse'][:num]

            with torch.no_grad():
                outputs = self.net.decode(embeddings_new)
                renders = self.gaussian_render.render_images(outputs['pointclouds'], cameras_row)
                renders = K.geometry.resize(renders, (H, W))

            return make_grid(renders, nrows=1, padval=padval, padsize=0)

        image_rows = []
        for i in range(num):
            image_rows.append(make_row(i))

        image_matrix = make_grid(image_rows, ncols=1, padval=0, padsize=0)

        if input_images is not None:
            top_row = np.hstack([np.zeros((H, W, 3), dtype=np.uint8),  make_grid(input_images[:num], nrows=1, padsize=0)])
            left_col = make_grid(input_images[:num], ncols=1, padsize=0)
            image_matrix = np.vstack([top_row, np.hstack([left_col, image_matrix])])

        # return cv2.resize(image_matrix, (1280, 1280))
        return image_matrix
