from __future__ import annotations
import os
import shutil
import time
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pims
import kornia
import pandas as pd
import tqdm
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from datasets.video_dataset import flip_pose_np
from tracking.alignment import CanonicalFaceAlignment
from tracking.landmarks import LandmarkDetector, VisionRunningMode, LandmarkDetectorFAN, LM68_RIGID_IDS, \
    flip_landmarks, convert_landmarks_mediapipe_to_dlib
# from tracking.local_features import LocalFeatureTracker
from tracking.optim import MultiViewOptimization
from tracking.video_processor import VideoProcessor
from utils.util import compute_camera_transform, homogeneous_points, transform_camera, transform_landmarks, \
    get_camera_azimuth, backproject_landmarks, get_images
from utils.nn import to_numpy
# from visualization.viewer import O3DSceneViewer, make_pointcloud

from visualization.vis import show_image, make_grid, add_landmarks_to_images, add_error_to_images
from visualization import vis

GREEN = (0, 1.0, 0)
RED = (0.0, 0.0, 1.0)
PINK = (1.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)


def find_face_masks(segs, num_erode_iters=5, device='cuda') -> (np.ndarray, np.ndarray):

    # all values above 110 (= neck) indicate face parts
    masks = segs > 115

    t = time.time()
    masks = torch.tensor(masks).unsqueeze(1).int()
    # kernel = torch.ones(5, 5).int().to(device)
    # for i in range(num_erode_iters):
    #     masks = kornia.morphology.erosion(masks, kernel)

    find_largest_component = False
    if find_largest_component:

        def find_connected_components(masks: torch.Tensor)-> torch.Tensor:
            inputs = masks.to(dtype=torch.float32)
            return kornia.contrib.connected_components(inputs, num_iterations=512)

        components = find_connected_components(masks)

        face_masks = torch.zeros_like(masks)
        for i in range(components.shape[0]):
            labels, counts  = torch.unique(components[i], sorted=True, return_counts=True)
            if len(counts) < 2:
                continue
            max_id = int(torch.argmax(counts[1:])) + 1  # first element is always background (=0)
            largest_component_pixels = components[i] == labels[max_id]
            face_masks[i].masked_fill_(largest_component_pixels, 1)
    else:
        face_masks = masks

    face_masks = face_masks.squeeze(1).bool()
    face_pixel_counts = torch.count_nonzero(face_masks, dim=(1, 2))
    # print(face_pixel_counts)

    # print("time erode: ", time.time()-t)
    # show_image("face masks", make_grid(face_masks, cmap='jet'), wait=0)

    return to_numpy(face_masks), to_numpy(face_pixel_counts)


class OfflineFaceTracker(VideoProcessor):
    def __init__(
            self,
            tasks: list[str],
            asset_dir: str = "./data",
            batch_size: int = 8,
            image_size: int = 512,
            fov: float = 30,
            **kwargs
    ):
        super().__init__(**kwargs)

        self._tasks = tasks
        self._image_size = image_size
        self._fov = fov

        if 'segment' in self._tasks:
            lib_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'libs')
            from libs.RobustVideoMatting.inference import Converter as FaceMatting
            self._matting = FaceMatting(
                variant='resnet50',
                checkpoint=os.path.join(lib_dir, 'RobustVideoMatting/rvm_resnet50.pth'),
                device=self._device,
            )
            from tracking.face_parsing import FaceParser, FACE_SURFACE_PARTS
            self._face_parser = FaceParser(
                checkpoint=os.path.join(lib_dir, 'face_parsing_pytorch/res/cp/79999_iter.pth'),
                batch_size=batch_size,
                progress=self._progress,
                device=self._device
            )

        if 'detect' in self._tasks:
            self._landmark_detector_mp = LandmarkDetector(
                asset_dir, running_mode='image', flip_input=True, min_confidence=0.0
            )
            self._landmark_detector_fan = LandmarkDetectorFAN(flip_input=True)

        # self._feature_tracker = LocalFeatureTracker(device=self._device)

        if 'align' in self._tasks:
            canonical_obj_fpath = os.path.join(asset_dir, "face_model_with_iris.obj")

            focal_length = 0.5 / np.tan(np.radians(fov/2))
            self._canonical_face_alignment = CanonicalFaceAlignment(
                canonical_obj_fpath,
                cam_pos=np.array([[0.5, -0.5, focal_length]]),
                look_at=np.array([[0.5, -0.5, 0.0]]),
                fov=fov
            )
            self.optimization = MultiViewOptimization(
                cam_canonical_init=self._canonical_face_alignment._cam,
                landmarks_canonical=self._canonical_face_alignment._lms_ref,
                progress=self._progress,
                device=self._device
            )

    def _save_camera(self, path, cam):
        # Save pose as 3x4 world-to-view matrix [R|T]
        np.savetxt(path, to_numpy(torch.hstack([cam.R[0], cam.T.T])), fmt='%.4f')

    def _save_cameras(self, filepaths, cameras):
        for path, cam in zip(filepaths, cameras):
            self._save_camera(path, cam)

    def _save_landmarks(self, filepaths, landmarks):
        for path, lms in zip(filepaths, landmarks):
            np.savetxt(path, to_numpy(lms), fmt='%.5f')

    def select_frames(
            self,
            video_path: str,
            interval: int = None,
            max_frame_count: int = None,
            max_clip_length: int = None,
    ):

        video, _ = get_images(video_path)

        # video = pims.PyAVVideoReader(video_path)

        if interval is None:
            interval = 1

        frame_ids = list(range(0, len(video), interval))

        if max_clip_length is not None:
            frame_ids = frame_ids[:max_clip_length]

        if max_frame_count is not None:
            frame_count = len(frame_ids)
            keep_idx = np.linspace(0, frame_count-1, num=min(frame_count, max_frame_count)).astype(int)
            frame_ids = [frame_ids[i] for i in keep_idx]

        return frame_ids

    def extract_images(
            self,
            video_path: str,
            output_dir: str,
            frame_ids: list[int],
            overwrite: bool = False,
            extension: str = '.jpg',
            output_width: int = 512,
            output_height: int = 512,
            keep_aspect=True
    ) -> (list[int], list[np.ndarray]):

        # video = pims.PyAVVideoReader(video_path)
        video, _ = get_images(video_path)

        # delete old data
        if overwrite and os.path.exists(output_dir) and os.path.isdir(output_dir) :
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        if frame_ids is None:
            frame_ids = list(range(len(video)))

        images = []

        for fid in tqdm.tqdm(frame_ids, disable=not self._progress):
            img_path = os.path.join(output_dir, self.format_frame_id(fid) + extension)
            source_image = video[fid][..., ::-1]
            dsize = (output_width, output_height)
            h, w = source_image.shape[:2]
            # margin_y, margin_x = 0, 0
            # if keep_aspect and w != h:
            #     aspect = float(w) / float(h)
            #     if w > h:
            #         new_width = int(output_width / aspect)
            #         image = cv2.resize(source_image, dsize=(new_width, 512))
            #         margin_x = (new_width - output_width) // 2
            #     else:
            #         new_height = int(output_height / aspect)
            #         image = cv2.resize(source_image, dsize=(512, new_height))
            #         margin_y = (new_height - output_width) // 2
            if keep_aspect and w != h:
                aspect = float(w) / float(h)
                if h > w:
                    new_width = int(output_height * aspect)
                    dsize = (new_width, output_height)
                else:
                    new_height = int(output_width * aspect)
                    dsize = (output_width, new_height)

            img_resized_bgr = cv2.resize(source_image, dsize=dsize, interpolation=cv2.INTER_LANCZOS4)
            # pad image if necessary
            # image = image[margin_y: margin_y + output_height, margin_x: margin_x + output_width]

            def pad_image(image, width, height):
                pad_y = (height - image.shape[0]) // 2
                pad_x = (width - image.shape[1]) // 2
                padding = [
                    (pad_y, height - (image.shape[0] + pad_y)),
                    (pad_x, width - (image.shape[1] + pad_x)),
                    (0, 0)
                ]
                return np.pad(image, pad_width=padding)

            img_resized_bgr = pad_image(img_resized_bgr, output_width, output_height)

            cv2.imwrite(img_path, img_resized_bgr)
            images.append(img_resized_bgr)

        return frame_ids, images


    def run_video_matting(self, input_dir: str, output_dir: str, overwrite: bool = True) -> None:
        if not os.path.exists(input_dir):
            print(f"No frames found at: {input_dir}!")
            return

        # delete old data
        if overwrite and os.path.exists(output_dir) and os.path.isdir(output_dir) :
            shutil.rmtree(output_dir)

        if not overwrite and os.path.exists(output_dir):
            return

        self._matting.convert(
            input_source=input_dir,
            input_resize=None,
            output_type='png_sequence',
            # output_composition=output_dir,
            output_alpha=output_dir,
            seq_chunk=self._batch_size,
            progress=self._progress
        )

    def run_face_parsing(self, input_dir: str, output_dir: str, overwrite: bool = True) -> None:
        if not os.path.exists(input_dir):
            print(f"Face parsing input dir {input_dir} does not exist. Please run video matting first.")
            return

        # delete old data
        if overwrite and os.path.exists(output_dir) and os.path.isdir(output_dir) :
            shutil.rmtree(output_dir)

        if not overwrite and os.path.exists(output_dir):
            return

        self._face_parser.evaluate(input_dir=input_dir, output_dir=output_dir)

    def get_images(self, input_path):
        frame_rate = 25
        if os.path.splitext(input_path)[1] == '.mp4':
            source_images = pims.PyAVVideoReader(input_path)
            frame_rate = source_images.frame_rate
        elif os.path.splitext(input_path)[1] in ['.png', '.jpg']:
            source_images = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        elif os.path.isdir(input_path):
            source_images = pims.ImageSequence(os.path.join(input_path, '*.jpg'))
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        if isinstance(source_images, np.ndarray) and len(source_images.shape) == 3:
            source_images = [source_images]

        return source_images, frame_rate

    def run_landmark_detection(self, input_path, output_dir, frame_ids = None, overwrite: bool = True,
                               show: bool = False, save_vis: bool = True):

        # delete old data
        if overwrite and os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        if os.path.exists(output_dir):
            return

        os.makedirs(output_dir, exist_ok=True)

        # images, frame_rate = self.get_images(input_path)
        images, frame_rate = get_images(input_path)

        if frame_ids is None:
            frame_ids = range(len(images))

        run_frame_ids = frame_ids

        # extract landmarks from all frames when using MediaPipe detector in video mode
        video_mode = self._landmark_detector_mp._mode == VisionRunningMode.VIDEO
        if video_mode:
            run_frame_ids = range(len(images))

        self._landmark_detector_mp.reset()

        for fid in tqdm.tqdm(run_frame_ids, disable=not self._progress):
            image = images[fid]
            frame_timestamp_ms = int(1000 * fid / frame_rate)
            landmarks_mp = self._landmark_detector_mp.detect(image, frame_timestamp_ms, show=show)

            if fid in frame_ids:

                path = os.path.join(output_dir, self.format_frame_id(fid) + '_landmarks.txt')
                np.savetxt(path, landmarks_mp, fmt='%.5f')

                if self._landmark_detector_fan is not None:
                    landmarks_fan, score_fan = self._landmark_detector_fan.detect(image, show=show)
                    if landmarks_fan is not None:
                        path = os.path.join(output_dir, self.format_frame_id(fid) + '_landmarks_fan.txt')
                        np.savetxt(path, landmarks_fan, fmt='%.5f')

                        if show:
                            landmarks_mp_68 = convert_landmarks_mediapipe_to_dlib(landmarks_mp)
                            vis_landmarks_mp = vis.draw_landmarks(image, landmarks_mp, color=(255, 255, 255))
                            vis_landmarks_mp_68 = vis.draw_landmarks(image, landmarks_mp_68, color=(255, 255, 255))
                            vis_landmarks_fan = vis.draw_landmarks(image, landmarks_fan, color=(255, 255, 255))
                            vis_lms = make_grid([image, vis_landmarks_mp, vis_landmarks_mp_68, vis_landmarks_fan])
                            show_image("landmarks mediapipe", vis_lms, wait=5)


    def run_face_alignment(
            self,
            landmark_dir: str,
            output_dir: str,
            overwrite: bool = True
    ):

        def align_mp_to_fan(lms_mp, lms_fan, M_fan_to_canonical):
            landmarks2d_mp_68 = convert_landmarks_mediapipe_to_dlib(lms_mp)
            M_transform = compute_camera_transform(lms_fan[LM68_RIGID_IDS], landmarks2d_mp_68[LM68_RIGID_IDS])
            M_mp_to_fan = np.linalg.inv(M_transform)
            M_mp_to_canonical = M_fan_to_canonical @ M_mp_to_fan
            lms_mp_fixed = transform_landmarks(lms_mp, M_mp_to_fan)
            lms_mp_aligned_fixed = transform_landmarks(lms_mp, M_mp_to_canonical)
            return lms_mp_fixed, lms_mp_aligned_fixed

        landmark_files = sorted(glob.glob(os.path.join(landmark_dir, '*_landmarks.txt')))
        landmark_files_fan = [f.replace('_landmarks', '_landmarks_fan') for f in landmark_files]
        landmark_aligned_files = [f.replace('_landmarks', '_landmarks_aligned') for f in landmark_files]
        landmark_aligned_files_fan = [f.replace('_landmarks', '_landmarks_aligned_fan') for f in landmark_files]
        pose_files = [f.replace('landmarks', 'pose') for f in landmark_files]
        transform_files = [f.replace('landmarks', 'transform') for f in landmark_files]

        for i in tqdm.tqdm(range(len(landmark_files)), disable=not self._progress):
            if os.path.isfile(pose_files[i]) and not overwrite:
                continue
            lms_mp = np.loadtxt(landmark_files[i])
            lms_mp[:, 1] *= -1
            lms_mp[:, 2] *= -1
            cam_new, lms_new_mp, M = self._canonical_face_alignment.align(lms_mp)

            debug_flips = False
            if debug_flips:
                asset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../assets")
                _mirror_ids = np.loadtxt(os.path.join(asset_dir, "mediapipe_mirror_ids_obj.txt")).astype(int)
                lms_2d_flip = flip_landmarks(lms_mp, mirror_ids=_mirror_ids)
                cam_new_flip, lms_new_mp_flip, M_flip = self._canonical_face_alignment.align(lms_2d_flip)

                # visualizer = O3DSceneViewer()
                # visualizer.add_model(make_pointcloud(lms_mp, colors=(0, 0, 1)))
                # visualizer.add_model(make_pointcloud(lms_2d_flip, colors=(1, 0, 1)))
                # visualizer.add_model(make_pointcloud(self._canonical_face_alignment._lms_ref, colors=(0.5, 0.5, 0.5)))
                # visualizer.add_model(make_pointcloud(lms_new_mp, colors=(0, 0, 1)))
                # visualizer.add_model(make_pointcloud(lms_new_mp_flip, colors=(1, 0, 1)))
                # visualizer.add_camera(self._canonical_face_alignment._cam, (0.5, 0.5, 0.5))
                # visualizer.add_camera(cam_new, (0, 0, 1))
                # visualizer.add_camera(cam_new_flip, (1, 0, 1))
                # visualizer.show()

            # cameras.append(cam_new)
            # landmarks_aligned.append(lms_new)
            # transforms.append(M)
            np.savetxt(landmark_aligned_files[i], to_numpy(lms_new_mp), fmt='%.5f')
            np.savetxt(transform_files[i], to_numpy(M), fmt='%.4f')
            np.savetxt(pose_files[i], to_numpy(torch.hstack([cam_new.R[0], cam_new.T.T])), fmt='%.4f')

            if os.path.isfile(landmark_files_fan[i]):
                lms_fan = np.loadtxt(landmark_files_fan[i])
                lms_fan[:, 1] *= -1
                cam_new_fan, lms_new_fan, M_fan = self._canonical_face_alignment.align(lms_fan)

                cam_distance = torch.linalg.norm(
                    cam_new.get_camera_center() - cam_new_fan.get_camera_center(), dim=1)

                azimuth = get_camera_azimuth(cam_new_fan)[0]
                # print(cam_distance.item(), azimuth)

                if cam_distance > 0.2 and abs(azimuth) > 60:
                    print(f"Fixing MP landmarks with FAN landmarks (cam_distance={cam_distance.item()})")
                    lms_mp_fixed, lms_mp_aligned_fixed = align_mp_to_fan(lms_mp, lms_fan, M_fan)
                    lms_mp_fixed[:, 1] *= -1
                    lms_mp_fixed[:, 2] *= -1
                    np.savetxt(landmark_files[i], to_numpy(lms_mp_fixed), fmt='%.5f')
                    np.savetxt(landmark_aligned_files[i], to_numpy(lms_mp_aligned_fixed), fmt='%.5f')

                np.savetxt(landmark_aligned_files_fan[i], to_numpy(lms_new_fan), fmt='%.5f')
                np.savetxt(transform_files[i], to_numpy(M_fan), fmt='%.4f')
                np.savetxt(pose_files[i], to_numpy(torch.hstack([cam_new_fan.R[0], cam_new_fan.T.T])), fmt='%.4f')

                if False:
                    visualizer = O3DSceneViewer()
                    visualizer.add_model(make_pointcloud(lms_mp, colors=(0, 0, 1)))
                    visualizer.add_model(make_pointcloud(lms_fan, colors=(1, 0, 1)))
                    visualizer.add_model(make_pointcloud(self._canonical_face_alignment._lms_ref, colors=(0.5, 0.5, 0.5)))
                    visualizer.add_model(make_pointcloud(lms_new_mp, colors=(0, 0, 1)))
                    visualizer.add_model(make_pointcloud(lms_new_fan, colors=(1, 0, 1)))
                    visualizer.add_model(make_pointcloud(lms_mp_fixed, colors=(0.4, 0.4, 1)))
                    visualizer.add_model(make_pointcloud(lms_mp_aligned_fixed, colors=(0.4, 0.4, 1)))
                    visualizer.add_camera(self._canonical_face_alignment._cam, (0.5, 0.5, 0.5))
                    visualizer.add_camera(cam_new, (0, 1, 0))
                    visualizer.add_camera(cam_new_fan, (1, 0, 1))
                    visualizer.show()


    def run_multiview_optimization(
            self,
            landmark_dir: str,
            pose_dir: str,
            frame_dir: str,
            frame_ids: list[int],
            overwrite: bool = False,
            show: bool = False
    ):

        def validate_frame_ids(frame_ids: list[int]) -> list[int]:
            def is_valid_id(fid):
                return os.path.isfile(self.get_path(landmark_dir, fid, '_landmarks_fan.txt'))
            return [fid for fid in frame_ids if is_valid_id(fid)]

        frame_ids = validate_frame_ids(frame_ids)

        if len(frame_ids) == 0:
            return

        landmark_files = [self.get_path(landmark_dir, i, '_landmarks.txt') for i in frame_ids]
        landmark_files_fan = [f.replace('_landmarks', '_landmarks_fan') for f in landmark_files]
        landmark_aligned_files = [f.replace('_landmarks', '_landmarks_aligned') for f in landmark_files]
        landmark_aligned_files_fan = [f.replace('_landmarks', '_landmarks_aligned_fan') for f in landmark_files]
        pose_files = [f.replace('landmarks', 'pose') for f in landmark_files]
        transform_files = [f.replace('landmarks', 'transform') for f in landmark_files]
        image_files = [os.path.join(frame_dir, self.format_frame_id(i) + '.jpg') for i in frame_ids]

        images = self._load_images(image_files=image_files)
        # images = pims.ImageSequence(os.path.join(frame_dir, '*.jpg'))

        landmarks_mp = self._load_landmarks(landmark_files)
        landmarks_fan = self._load_landmarks(landmark_files_fan)
        landmarks_mp_aligned = self._load_landmarks(landmark_aligned_files)
        landmarks3d_fan =  self._load_landmarks(landmark_aligned_files_fan)

        poses = np.array([np.loadtxt(pose_file) for pose_file in pose_files])
        R = poses[:, :3, :3]
        T = poses[:, :3, 3]
        cameras = FoVPerspectiveCameras(R=torch.tensor(R), T=torch.tensor(T), fov=self._fov)

        M_fan_to_canonical = np.array([np.loadtxt(tfile) for tfile in transform_files])

        h, w, _ = images[0].shape
        landmarks2d_mp = landmarks_mp[:, :, :2] * np.array([w, h])
        landmarks2d_fan = landmarks_fan[:, :, :2] * np.array([w, h])

        results = self.optimization.optimize(
            cameras=cameras,
            transforms=M_fan_to_canonical,
            landmarks=landmarks2d_fan,
            landmarks_aligned=landmarks3d_fan,
            landmarks_dense=landmarks2d_mp,
            landmarks_dense_aligned=landmarks_mp_aligned,
            images=images,
            num_iterations=50,
            frame_ids=frame_ids,
            show=show
        )

        landmarks_fan_final = results['landmarks']
        landmarks_mp_final = results['landmarks_dense']
        cameras_final = results['cameras']

        # disp_reg, _ = self.optimization.visualize(images, landmarks_mp_final, landmarks_mp, cameras)
        # show_image("mediapipe final", disp_reg, wait=0)

        # visualizer = O3DSceneViewer()
        # visualizer.add_model(make_pointcloud(landmarks_mp_final[0], colors=(0, 0, 1)))
        # visualizer.add_model(make_pointcloud(landmarks_fan_final[0], colors=(1, 0, 1)))
        # visualizer.add_camera(cameras[0], (0.5, 0.5, 0.5))
        # visualizer.add_camera(cameras_final[0], (1, 0, 1))
        # visualizer.show()

        # optimization.optimize_features(features, tracks, cameras, num_iterations=20000, show=show, landmarks3d=landmarks3d)

        self._save_cameras(pose_files, cameras_final)
        self._save_landmarks(landmark_aligned_files, landmarks_mp_final)
        self._save_landmarks(landmark_aligned_files_fan, landmarks_fan_final)


    def collect_metadata(
            self,
            metadata_dir: str,
            frame_dir: str,
            alpha_dir: str,
            seg_dir: str,
            landmark_dir: str,
            pose_dir: str,
            clip_name: str,
            clip_id: int,
            fov: float,
            overwrite: bool = False,
            show: bool = False,
            wait: bool = False,
    ):
        frame_list = []

        output_file = os.path.join(metadata_dir, 'frame_meta.csv')
        if os.path.isfile(output_file) and not overwrite:
            return frame_list

        pose_files = sorted(glob.glob(os.path.join(pose_dir, '*_pose.txt')))
        frame_ids = [int(self._parse_image_path(f)[1]) for f in pose_files]

        image_files = [self.get_path(frame_dir, i, '.jpg') for i in frame_ids]
        alpha_files = [self.get_path(alpha_dir, i, '.jpg') for i in frame_ids]
        seg_files = [self.get_path(seg_dir, i, '.png') for i in frame_ids]
        landmark_files = [self.get_path(landmark_dir, i, '_landmarks.txt') for i in frame_ids]
        aligned_landmark_files = [self.get_path(landmark_dir, i, '_landmarks_aligned.txt') for i in frame_ids]

        assert len(image_files) == len(landmark_files) == len(seg_files) == len(pose_files) == len(alpha_files)

        if len(image_files) == 0:
            print(f"No frames found at {frame_dir}")
            return frame_list

        os.makedirs(os.path.split(output_file)[0], exist_ok=True)

        images = self._load_images(image_files=image_files)
        B, H, W, C = images.shape

        segs = self._load_images(image_files=seg_files)[..., 0]
        alphas = self._load_images(image_files=alpha_files)
        landmarks = self._load_landmarks(landmark_files)[..., :2] * np.array((H, W))
        landmarks_aligned = self._load_landmarks(aligned_landmark_files)
        poses = self._load_poses(pose_files)

        assert len(images) == len(image_files)
        assert len(images) == len(landmarks) == len(segs) == len(poses) == len(alphas)

        masks = alphas > 180
        masks[segs == 0] = 0
        masks[segs == 90] = 0

        masked_images = images.copy()
        masked_images[~masks] = 0

        #
        # check camera pose estimation
        #

        # project 3D landmarks from first frame onto each image to check estimated camera poses
        ref_lms_3d = landmarks_aligned[0]
        cameras = FoVPerspectiveCameras(R=poses[:, :3, :3], T=poses[:, :3, 3], fov=fov)
        landmarks_proj = [backproject_landmarks(ref_lms_3d, cameras[i], W, H) for i in range(B)]
        # print(cameras.get_camera_center())

        camera_positions = cameras.get_camera_center()
        camera_distances = torch.linalg.norm(camera_positions, dim=1)

        # compute landmark reprojection error

        def mean_euclidean_dist(lms1, lms2):
            return (((lms1-lms2) ** 2).sum(axis=2) ** 0.5).mean(axis=1)

        reproj_errors = mean_euclidean_dist(landmarks, landmarks_proj)

        #
        # check segmentation
        #

        face_masks, face_pixels_counts = find_face_masks(segs)
        face_area = face_pixels_counts / (W * H)

        non_zero_counts = masks.reshape(len(masks), -1).sum(axis=1)
        face_area[non_zero_counts < 10000] = 0

        #
        # check luminance
        #

        def compute_luminance(images: np.ndarray, masks: np.ndarray, device='cpu') -> np.ndarray:
            # kornia rgb_to_luv expects float tensors of shape [B, 3, H, W] in range [0, 1]
            inputs = torch.tensor(images).float().permute(0, 3, 1, 2).to(device) / 255.0
            lum = to_numpy(kornia.color.rgb_to_luv(inputs)[:, 0])
            lum_per_image = np.array([lum[i, masks[i]].mean() for i in range(len(masks))])
            return lum_per_image

        # def compute_brightness(images: np.ndarray, masks: np.ndarray, device='cpu') -> np.ndarray:
        #     gray = torch.tensor(images).float().mean(dim=3)
        #     brightness_per_image = [gray[i, masks[i]].mean().item() for i in range(len(masks))]
        #     return np.array(brightness_per_image)
        def compute_brightness(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
            gray = images.astype(np.float32).mean(axis=3)
            brightness_per_image = [gray[i, masks[i]].mean() for i in range(len(masks))]
            return np.array(brightness_per_image)

        # t = time.time()
        # face_luminances = compute_luminance(images, masks)
        face_luminances = compute_brightness(images, masks)
        # print("time brightness: ", time.time()-t)
        # print(face_luminances)

        #
        # write results to CSV
        #

        def frame_id(path):
            return int(os.path.splitext(os.path.split(path)[1])[0].split('_')[0])

        for i in range(B):
            frame_list.append(dict(
                clip_name=clip_name,
                clip_id=clip_id,
                frame_id=frame_id(image_files[i]),
                face_area=face_area[i],
                luminance=face_luminances[i],
                camera_x=camera_positions[i, 0].item(),
                camera_y=camera_positions[i, 1].item(),
                camera_z=camera_positions[i, 2].item(),
                camera_distance=camera_distances[i].item(),
                reproj_error=reproj_errors[i],
                nonzero_count=non_zero_counts[i]
                # keypoints=landmarks[i]
            ))

        df = pd.DataFrame(frame_list)
        df.to_csv(output_file, float_format="%.4f", index=False)

        face_area_ok = face_area > 0.04
        luminance_ok = face_luminances > 10.
        camera_pose_ok = camera_distances < 3.0

        if not face_area_ok.all():
            print("Invalid face segmentation detected!")
            print(face_area)

        if not luminance_ok.all():
            print("Dark frame detected!")
            print(face_luminances)

        if not camera_pose_ok.all():
            print("Invalid camera pose detected!")
            print(to_numpy(camera_distances))

        bad_frame_detected = not (face_area_ok.all() and luminance_ok.all() and camera_pose_ok.all())

        if show:
            print(face_luminances.mean())
            ncols = 16

            image_grid = make_grid(images, ncols=ncols)
            alpha_grid = make_grid(alphas, ncols=ncols)
            seg_grid = make_grid(segs, ncols=ncols, cmap='jet')
            mask_grid = make_grid(masks, ncols=ncols)
            masked_image_grid = make_grid(masked_images, ncols=ncols)

            # cv2.imshow("images", cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR))
            # show_image("masks", mask_grid, wait=0)
            # show_image("alphas", alpha_grid)
            # cv2.imshow("alphas", cv2.cvtColor(alpha_grid, cv2.COLOR_RGB2BGR))
            images_with_landmarks = add_landmarks_to_images(images, landmarks, color=(255, 255, 255))
            images_with_projected_landmarks = add_landmarks_to_images(images, landmarks_proj, color=(255, 0, 255))
            images_with_both_landmarks = add_landmarks_to_images(images_with_landmarks, landmarks_proj, color=(255, 0, 255))
            images_with_projected_landmarks = add_error_to_images(images_with_projected_landmarks, camera_distances,
                                                                  loc='tr', vmax=10.0, size=1.0, thickness=2)
            images_with_projected_landmarks = add_error_to_images(images_with_projected_landmarks, reproj_errors,
                                                                  loc='tr+1', vmax=30.0, size=1.0, thickness=2)

            disp_image = make_grid([
                image_grid,
                alpha_grid,
                seg_grid,
                # make_grid(face_masks, ncols=ncols),
                mask_grid,
                masked_image_grid,
            ], ncols=1, padsize=0)

            show_image("segmentation", disp_image, f=0.4)

            disp_registration = make_grid([
                make_grid(images_with_landmarks, ncols=ncols),
                make_grid(images_with_projected_landmarks, ncols=ncols),
                make_grid(images_with_both_landmarks, ncols=ncols)
            ], ncols=1, padsize=0)

            show_image("registration", disp_registration, f=0.5)

            delay_ms = 0 if wait or bad_frame_detected else 5
            cv2.waitKey(delay_ms)

        return frame_list

    def track_video(
            self,
            video_path,
            output_dir: str = "./results",
            output_sub_folder: str = None,
            interval: int = None,
            max_frame_count: int = None,
            max_clip_length: int = None,
            clip_name: str | None = None,
            clip_id: int | str | None = None,
            overwrite: bool = False,
            show: bool = False,
            wait: bool = False
    ):
        if clip_name is None:
            clip_name = os.path.splitext(os.path.split(video_path)[1])[0]

        if clip_id is None:
            clip_id = clip_name

        if output_sub_folder is None:
            output_sub_folder = ""

        frame_dir = os.path.join(output_dir, 'frames', output_sub_folder)
        alpha_dir = os.path.join(output_dir, 'alpha', output_sub_folder)
        seg_dir = os.path.join(output_dir, 'seg_mask', output_sub_folder)
        pose_dir = os.path.join(output_dir, 'poses', output_sub_folder)
        metadata_dir = os.path.join(output_dir, 'metadata', output_sub_folder)
        image_dir = os.path.join(output_dir, 'images', output_sub_folder)
        landmark_dir = pose_dir

        frame_ids = self.select_frames(video_path, interval, max_frame_count, max_clip_length)

        _, images = self.extract_images(video_path, frame_dir, frame_ids, overwrite=overwrite)

        if 'segment' in self._tasks:
            self.run_video_matting(frame_dir, alpha_dir, overwrite=overwrite)
            self.run_face_parsing(frame_dir, seg_dir, overwrite=overwrite)

        save_masked_images = False
        if save_masked_images:
            def save_segmented_images(images, alpha_dir, output_dir, frame_ids, overwrite):
                if overwrite and os.path.exists(output_dir) and os.path.isdir(output_dir):
                    shutil.rmtree(output_dir)
                alphas, _ = get_images(alpha_dir)
                alphas = [cv2.resize(a, dsize=(images[0].shape[1], images[0].shape[0])) for a  in alphas]
                masked_images = [a[...,np.newaxis].astype(np.float32)/255.0 * im.astype(np.float32)/255.0
                                 for a, im in zip(alphas, images)]
                os.makedirs(image_dir, exist_ok=True)
                for fid, im in zip(frame_ids, masked_images):
                    img_path = os.path.join(output_dir, self.format_frame_id(fid) + '.jpg')
                    cv2.imwrite(img_path, (im*255.0).astype(np.uint8))

            save_segmented_images(images=images, alpha_dir=alpha_dir, output_dir=image_dir,
                                  frame_ids=frame_ids, overwrite=overwrite)

        if 'detect' in self._tasks:
            self.run_landmark_detection(video_path, landmark_dir, frame_ids, overwrite=overwrite, show=show)

        if 'align' in self._tasks:
            self.run_face_alignment(landmark_dir, pose_dir, overwrite=overwrite)

            self.run_multiview_optimization(
                landmark_dir=landmark_dir,
                pose_dir=pose_dir,
                frame_dir=frame_dir,
                frame_ids=frame_ids,
                overwrite=overwrite,
                show=show
            )

        if 'metadata' in self._tasks:
            self.collect_metadata(
                metadata_dir=metadata_dir,
                frame_dir=frame_dir,
                alpha_dir=alpha_dir,
                seg_dir=seg_dir,
                landmark_dir=landmark_dir,
                pose_dir=pose_dir,
                clip_name=clip_name,
                clip_id=clip_id,
                fov=self._fov,
                overwrite=overwrite,
                show=show,
                wait=wait
            )