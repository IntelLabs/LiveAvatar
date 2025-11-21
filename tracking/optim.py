import tqdm
import torch
import torch.nn.functional as F
import pytorch3d
# from pytorch3d.loss import chamfer_distance
from pytorch3d.renderer import FoVPerspectiveCameras
import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import cm

import visualization.vis
from tracking.landmarks import LM68_RIGID_IDS, lm478_to_lm68, tensor_lm478_to_lm68
from utils.util import transform_camera, transform_landmarks_torch, stack_cameras, get_camera_azimuth, \
    backproject_landmarks
from utils.nn import to_numpy
# from visualization.viewer import O3DSceneViewer, pairwise_backproject_landmarks, make_pointcloud
from visualization.vis import overlay_patches, show_image, draw_landmarks, draw_projected_landmarks, \
    make_grid, color_map

GREEN = (0, 1.0, 0)


def create_samples(N):
    samples = []
    for i in range(N)[::1]:
        for j in range(N)[::1]:
            if j == i:
                continue
            samples.append((i, j))
    return samples


class Model(torch.nn.Module):
    def __init__(self, landmarks_aligned, landmarks_dense_aligned, transforms, cam_init, device='cuda'):
        super().__init__()
        self.landmarks_aligned_init = landmarks_aligned.clone()
        self.landmarks_dense_aligned_init = landmarks_dense_aligned.clone()
        self.transforms = [torch.tensor(transforms[i]).float().to(device) for i in range(len(transforms))]
        self.cam_canonical_init = cam_init.clone()

        N = len(landmarks_aligned)

        rot0: torch.Tensor = pytorch3d.transforms.matrix_to_rotation_6d(
            torch.eye(3).unsqueeze(0).repeat(N, 1, 1)
        )

        self.rot = torch.nn.Parameter(rot0, requires_grad=True)
        self.T = torch.nn.Parameter(torch.zeros(N, 3), requires_grad=True)
        self.fov = torch.nn.Parameter(torch.ones(1) * cam_init.fov.item(), requires_grad=False)
        self.landmarks_aligned_opt = torch.nn.Parameter(landmarks_aligned, requires_grad=True)
        self.landmarks_dense_aligned_opt = torch.nn.Parameter(landmarks_dense_aligned, requires_grad=True)

    def get_new_cameras(self, rot6d: torch.Tensor, translation: torch.Tensor):
        cameras_new = []
        R = pytorch3d.transforms.rotation_6d_to_matrix(rot6d)
        for i in range(rot6d.shape[0]):
            M = torch.hstack([R[i], translation[i].unsqueeze(1)])
            M = M @ self.transforms[i]
            cam_new = transform_camera(self.cam_canonical_init, M.unsqueeze(0), self.fov)
            cameras_new.append(cam_new)
        return cameras_new

    def forward(self):
        landmarks_aligned_new = torch.zeros_like(self.landmarks_aligned_opt)
        landmarks_aligned_rigid = torch.zeros_like(self.landmarks_aligned_opt)
        landmarks_dense_aligned_new = torch.zeros_like(self.landmarks_dense_aligned_opt)
        landmarks_dense_aligned_rigid = torch.zeros_like(self.landmarks_dense_aligned_opt)

        R = pytorch3d.transforms.rotation_6d_to_matrix(self.rot)
        transforms_opt = torch.cat([R, self.T.unsqueeze(2)], dim=2)

        for i in range(R.shape[0]):
            M = torch.hstack([R[i], self.T[i].unsqueeze(1)])
            landmarks_aligned_new[i] = transform_landmarks_torch(self.landmarks_aligned_opt[i], M.unsqueeze(0))
            landmarks_aligned_rigid[i] = transform_landmarks_torch(self.landmarks_aligned_init[i], M.unsqueeze(0))
            landmarks_dense_aligned_new[i] = transform_landmarks_torch(self.landmarks_dense_aligned_opt[i], M.unsqueeze(0))
            landmarks_dense_aligned_rigid[i] = transform_landmarks_torch(self.landmarks_dense_aligned_init[i], M.unsqueeze(0))

        cameras_new = self.get_new_cameras(self.rot, self.T)
        return (stack_cameras(cameras_new), transforms_opt,
                landmarks_aligned_new, landmarks_aligned_rigid,
                landmarks_dense_aligned_new, landmarks_dense_aligned_rigid)


class MultiViewOptimization():

    def __init__(self,
                 cam_canonical_init: FoVPerspectiveCameras,
                 landmarks_canonical: np.ndarray,
                 progress: bool = True,
                 device='cuda'):
        self.cam_canonical_init = cam_canonical_init.clone().to(device)
        self.progress = progress
        self.device = device
        self.lms68_canonical = torch.tensor(landmarks_canonical).to(device)

    def draw_projected_patches(self, images, target_id, proj_ids, cameras, landmarks, k=9):
        target_image = images[target_id]
        image_patches = target_image.copy()
        h, w, _ = image_patches.shape
        for i in proj_ids:
            image_patches = overlay_patches(
                image_patches,
                images[i],
                backproject_landmarks(landmarks[i], cameras[target_id], w, h),
                backproject_landmarks(landmarks[i], cameras[i], w, h),
                k=k,
                alpha=0.2
            )
        return image_patches

    def get_colors(self, num, cmap: str | mpl.colors.Colormap = cm.plasma):
        colors = cm.get_cmap(cmap)(np.linspace(0, 1, num))
        return [mpl.colors.to_rgb(c) for c in colors]

    def visualize(self, images, landmarks_aligned, landmarks2d, cameras, frame_ids=None, ncols=16):
        N = min(len(landmarks_aligned), len(cameras))
        colors = self.get_colors(N)
        h, w, _ = images[0].shape

        images_with_landmarks = [
            draw_landmarks(images[i], landmarks2d[i], color=(255, 255, 255)) for i in range(N)
        ]
        landmarks_proj = [
            to_numpy(backproject_landmarks(landmarks_aligned[i], cameras[i], w, h)) for i in range(N)
        ]
        # images_with_landmarks = add_landmarks_to_images(images_with_landmarks, landmarks_proj, color=(255, 0, 255))

        images_with_landmarks = [
            draw_landmarks(img, lms, color=(255, 0, 255)) for img, lms in zip(images_with_landmarks, landmarks_proj)
        ]

        if frame_ids is not None:
            images_with_landmarks = visualization.vis.add_label_to_images(images_with_landmarks, frame_ids)

        images_with_projected_landmarks = []
        for target_id in range(N):
            image_lms = draw_landmarks(images[target_id], landmarks2d[target_id])
            for source_id in range(N):
                # if source_id == target_id:
                #     continue
                image_lms = draw_projected_landmarks(image_lms, landmarks_aligned[source_id], cameras[target_id], color=colors[source_id])
            images_with_projected_landmarks.append(image_lms)

        show_patches = False
        image_patches = None
        if show_patches:
            image_patches = self.draw_projected_patches(images, target_id=0, proj_ids=range(3, N), cameras=cameras,
                                                        landmarks=landmarks_aligned, k=9)
            images_with_patches = []
            for target_id in range(N):
                img = self.draw_projected_patches(
                    images=images, target_id=target_id, proj_ids=[0], cameras=cameras,
                    landmarks=landmarks_aligned, k=9
                )
                images_with_patches.append(img)

        disp_registration = make_grid([
            make_grid(images_with_landmarks, ncols=ncols),
            make_grid(images_with_projected_landmarks, ncols=ncols),
            # make_grid(images_with_patches, ncols=ncols)
        ], ncols=1, padsize=0)
        return disp_registration, image_patches

    def visualize_reconstruction(self, images, track_keypoints, track_mat_idx, track_mat_2d, cameras, ncols=16):

        N = min(track_mat_idx.shape[1], len(cameras))
        h, w, _ = images[0].shape

        keypoints3d = [track_keypoints[track_mat_idx[:, fid] >= 0].cpu() for fid in range(N)]
        keypoints2d = [track_mat_2d[track_mat_idx[:, fid] >= 0, fid] for fid in range(N)]

        images_with_landmarks = [
            draw_landmarks(images[i], keypoints2d[i], color=(255, 255, 255)) for i in range(N)
        ]
        landmarks_proj = [
            to_numpy(backproject_landmarks(keypoints3d[i], cameras[i], w, h)) for i in range(N)
        ]
        images_with_landmarks = [
            draw_landmarks(img, lms, color=(255, 0, 255)) for img, lms in zip(images_with_landmarks, landmarks_proj)
        ]

        disp_registration = make_grid([
            make_grid(images_with_landmarks, ncols=ncols),
        ], ncols=1, padsize=0)
        return disp_registration #, image_patches

    def optimize(
            self,
            cameras: FoVPerspectiveCameras | list[FoVPerspectiveCameras],
            transforms: list[np.ndarray] | np.ndarray,
            landmarks: np.ndarray,
            landmarks_aligned: np.ndarray,
            landmarks_dense: np.ndarray,
            landmarks_dense_aligned: np.ndarray,
            images,
            num_iterations=20,
            frame_ids=None,
            show=False
    ):
        np.set_printoptions(precision=3, suppress=True)

        assert len(landmarks) == len(images) == len(landmarks_aligned)
        N, K, _ = landmarks.shape
        h, w, _ = images[0].shape
        device = self.device

        if show and False:
            disp_reg, disp_patches = self.visualize(images, landmarks_aligned, landmarks, cameras)
            # show_image("patches init", disp_patches)
            show_image("registration init", disp_reg, f=0.6, wait=1)

            if False:
                visualizer = O3DSceneViewer()
                colors = self.get_colors(N)
                for fid in range(N):
                    # pc_lms = make_pointcloud(landmarks[fid], colors=colors[fid])
                    # visualizer.add_model(pc_lms)
                    pc_lms = make_pointcloud(landmarks_aligned[fid], colors=colors[fid])
                    visualizer.add_model(pc_lms)
                    visualizer.add_camera(cameras[fid], (0.5, 0.5, 0.5))
                visualizer.show()

        landmarks = torch.tensor(landmarks, dtype=torch.float32, device=device)[..., :2]
        landmarks_aligned = torch.tensor(landmarks_aligned, dtype=torch.float32, device=device)
        landmarks_dense = torch.tensor(landmarks_dense, dtype=torch.float32, device=device)[..., :2]
        landmarks_dense_aligned = torch.tensor(landmarks_dense_aligned, dtype=torch.float32, device=device)

        model = Model(
            landmarks_aligned,
            landmarks_dense_aligned,
            transforms,
            cam_init=self.cam_canonical_init,
            device=device
        ).to(device)

        optimizer = torch.optim.Adam([
            model.rot,
            model.T,
            model.landmarks_aligned_opt,
            model.landmarks_dense_aligned_opt
        ], lr=0.001)
        optimizer_fov = torch.optim.Adam([model.fov], lr=1.0)

        landmarks_target = landmarks.repeat_interleave(N, dim=0)
        landmarks_dense_target = landmarks_dense.repeat_interleave(N, dim=0)
        canonical_target = self.lms68_canonical.unsqueeze(0).repeat(N, 1, 1)

        rigid_weight = torch.ones(K) * 0.1
        rigid_weight[LM68_RIGID_IDS] = 1.0
        rigid_weight = rigid_weight.reshape(1, -1, 1).to(device)

        azimuths = get_camera_azimuth(cameras)
        frontals = abs(azimuths) < 60
        num_frontals = np.count_nonzero(frontals)
        landmarks_dense_target_frontal = landmarks_dense[frontals].repeat_interleave(num_frontals, dim=0)

        for iter in tqdm.tqdm(range(num_iterations), disable=not self.progress):
            optimizer.zero_grad()
            optimizer_fov.zero_grad()

            (cameras_new, _, landmarks_aligned_new, landmarks_aligned_rigid,
             landmarks_dense_aligned_new, landmarks_dense_aligned_rigid) = model()

            loss_deform = F.l1_loss(landmarks_aligned_new, landmarks_aligned_rigid) * 1000

            loss_align = F.pdist(landmarks_aligned_new.reshape(N, -1), p=1).mean() * 0.1

            # lms_proj_ij = (landmarks_aligned_new[:, rigid_ids], cameras_new, w, h)
            # loss_proj = F.mse_loss(lms_proj_ij, landmarks_target[:, rigid_ids]) * 1
            lms_proj_ij = pairwise_backproject_landmarks(landmarks_aligned_new, cameras_new, w, h)
            loss_proj = F.mse_loss(lms_proj_ij * rigid_weight, landmarks_target * rigid_weight) * 1

            # lms_proj_ij = pairwise_backproject_landmarks(landmarks_aligned_new, cameras_new, w, h)
            diag_ids = range(0, len(landmarks_target), N)
            loss_reproj = F.mse_loss(lms_proj_ij[diag_ids], landmarks_target[diag_ids])

            loss = loss_proj + loss_align + loss_deform + loss_reproj
            if not self.progress and show and ((iter+1) % 1 == 0 or (iter == 0)):
                print(f"[{iter+1}/{num_iterations}] "
                      f"loss={loss.item():.4f} align={loss_align.item():.4f} deform={loss_deform.item():.4f} proj={loss_proj.item():.4f}  reproj={loss_reproj.item():.4f} "
                      f"T={to_numpy(model.T[0])} rot={to_numpy(model.rot[0])} fov={to_numpy(model.fov)}")

            if True:
                # loss_align = F.pdist(landmarks_dense_aligned_new[:, LM68_RIGID_IDS].reshape(len(LM68_RIGID_IDS), -1), p=1).mean() * 0.1
                loss_align = F.pdist(landmarks_dense_aligned_new.reshape(N, -1), p=1).mean() * 1.0

                loss_deform = torch.zeros(1, requires_grad=True, device=self.device)
                loss_proj = torch.zeros(1, requires_grad=True, device=self.device)
                loss_reproj = torch.zeros(1, requires_grad=True, device=self.device)

                if num_frontals > 0:
                    loss_deform = F.l1_loss(landmarks_dense_aligned_new[frontals], landmarks_dense_aligned_rigid[frontals]) * 100

                    cameras_new_frontal = stack_cameras([cameras_new[int(i)] for i in np.where(frontals)[0]])
                    lms_proj_ij = pairwise_backproject_landmarks(landmarks_dense_aligned_new[frontals], cameras_new_frontal, w, h)
                    loss_proj = F.mse_loss(lms_proj_ij, landmarks_dense_target_frontal)

                    lms_reproj = backproject_landmarks(landmarks_dense_aligned_new, cameras_new, w, h)
                    loss_reproj = F.mse_loss(lms_reproj[frontals], landmarks_dense[frontals])

                # loss_chamfer = chamfer_distance(
                #     tensor_lm478_to_lm68(landmarks_dense_aligned_new),
                #     landmarks_aligned_new,
                #     single_directional=True
                # )[0] * 10000 * 1

                landmarks_dense_aligned_new_68 = tensor_lm478_to_lm68(landmarks_dense_aligned_new)
                loss_chamfer = F.mse_loss(landmarks_dense_aligned_new_68, landmarks_aligned_new) * 100000

                loss_canonical = F.mse_loss(landmarks_dense_aligned_new_68, canonical_target) * 100

                loss =  loss + loss_align + loss_deform + loss_proj + loss_reproj + loss_chamfer + loss_canonical
                if not self.progress and show and ((iter+1) % 1 == 0 or (iter == 0)):
                    print(f"[{iter+1}/{num_iterations}] "
                          f"loss={loss.item():.4f} align={loss_align.item():.4f} deform={loss_deform.item():.4f} proj={loss_proj.item():.4f} reproj={loss_reproj.item():.4f} chamfer={loss_chamfer.item():.4f} canon={loss_canonical.item():.4f}"
                          f"T={to_numpy(model.T[0])} rot={to_numpy(model.rot[0])} fov={to_numpy(model.fov)}")

            loss.backward()
            optimizer.step()
            optimizer_fov.step()

            if (iter+1) % 50 == 0 and iter >= 0 and show:

                disp_reg, disp_patches = self.visualize(images, landmarks_aligned_new, landmarks, cameras_new, frame_ids)
                # show_image("patches opt", disp_patches)
                show_image("registration opt", disp_reg, f=1.0, wait=5)

                disp_reg, disp_patches = self.visualize(images, landmarks_dense_aligned_new, landmarks_dense, cameras_new, frame_ids)
                # show_image("patches opt", disp_patches)
                show_image("registration opt dense", disp_reg, f=1.0, wait=0)

                if False:
                    visualizer = O3DSceneViewer()
                    colors = self.get_colors(N)
                    for fid in range(N):
                        pc_lms = make_pointcloud(landmarks_aligned_new[fid], colors=colors[fid])
                        visualizer.add_model(pc_lms)
                        visualizer.add_camera(cameras[fid], (0.5, 0.5, 0.5))
                        visualizer.add_camera(cameras_new[fid], colors[fid])
                        visualizer.add_model(make_pointcloud(self.lms68_canonical, colors=(0, 1, 0)))
                    visualizer.show()

        cameras_final, transforms_opt, landmarks_aligned_final, _, landmarks_dense_aligned_final, _ = model()

        results = dict(
            cameras=[cam.cpu() for cam in cameras_final],
            M_opt=to_numpy(transforms_opt),
            landmarks=to_numpy(landmarks_aligned_final),
            landmarks_dense=to_numpy(landmarks_dense_aligned_final),
        )

        return results
