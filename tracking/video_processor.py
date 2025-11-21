from __future__ import annotations
import os
import numpy as np
import cv2
import glob


def is_color(img):
    return len(img.shape) == 3 and img.shape[2] == 3


class VideoProcessor():
    def __init__(
            self,
            batch_size: int = 8,
            progress: bool = True,
            device: str='cuda'
    ):
        self._image_size = 512
        self._batch_size = batch_size
        self._device = device
        self._progress = progress

    def get_frame_id(self, landmark_path: str) -> int:
        filename = os.path.split(landmark_path)[1]
        return int(os.path.splitext(filename)[0].split('_')[0])

    def format_frame_id(self, frame_id: int) -> str:
        return str(frame_id).zfill(4)

    def _load_images(
            self,
            clip_name: str | None = None,
            image_files: list[str] = None,
            interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        assert clip_name is not None or image_files is not None

        if image_files is None:
            image_files = sorted(glob.glob(os.path.join(self._image_root, clip_name, '*.jpg')))

        def to_rgb(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_color(img) else img

        images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in image_files]
        images = [to_rgb(img) for img in images]
        images = [cv2.resize(img, dsize=(self._image_size, self._image_size), interpolation=interpolation) for img in images]
        return np.array(images)  # RGB

    def load_images_from_video(self, video_path: str) -> list[np.ndarray]:
        images_bgr = []
        clip_raw = cv2.VideoCapture(video_path)
        while clip_raw.isOpened():
            ret, frame = clip_raw.read()
            if not ret:
                break
            images_bgr.append(frame)

        # clip_raw = VideoFileClip(video_path)
        # images_bgr = [clip_raw[i][...,::-1] for i in range(clip_raw.n_frames)]
        images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]
        return images_rgb

    def get_path(self, base_dir: str, i: int, fname_suffix: str):
        return os.path.join(base_dir, self.format_frame_id(i) + fname_suffix)

    def _parse_image_path(self, img_path: str) -> (str, str):
        basename, filename = os.path.split(img_path)
        clip_name = os.path.split(basename)[1]
        frame_id = os.path.splitext(filename)[0].split('_')[0]
        return clip_name, frame_id

    def get_image_path(self, clip_name, frame_id: int) -> str:
        return os.path.join(self._image_root, clip_name, self.format_frame_id(frame_id) + '.jpg')

    # def get_landmark_path(self, clip_name, frame_id: int) -> str:
    #     return os.path.join(self._pose_root, clip_name, self.format_frame_id(frame_id) + '_landmarks.txt')

    def get_landmark_path(self, landmark_dir, frame_id: int) -> str:
        return os.path.join(landmark_dir, self.format_frame_id(frame_id) + '_landmarks.txt')

    def get_aligned_landmark_path(self, clip_name, frame_id: int) -> str:
        return os.path.join(self._pose_root, clip_name, self.format_frame_id(frame_id) + '_landmarks_aligned.txt')

    def get_alpha_path(self, clip_name, frame_id: int) -> str:
        return os.path.join(self._alpha_root, clip_name, self.format_frame_id(frame_id) + '.jpg')

    def get_seg_path(self, clip_name, frame_id: int) -> str:
        return os.path.join(self._seg_root, clip_name, self.format_frame_id(frame_id) + '.png')

    def _to_landmark_path(self, img_path: str):
        clip_name, frame_id = self._parse_image_path(img_path)
        return os.path.join(self._pose_root, clip_name, frame_id + '_landmarks.txt')

    def _to_aligned_landmark_path(self, img_path: str):
        clip_name, frame_id = self._parse_image_path(img_path)
        return os.path.join(self._pose_root, clip_name, frame_id + '_landmarks_aligned.txt')

    def _to_pose_path(self, img_path: str):
        clip_name, frame_id = self._parse_image_path(img_path)
        return os.path.join(self._pose_root, clip_name, frame_id + '_pose.txt')

    def _to_alpha_path(self, img_path: str):
        clip_name, frame_id = self._parse_image_path(img_path)
        return os.path.join(self._alpha_root, clip_name, frame_id + '.jpg')

    def _to_seg_path(self, img_path: str):
        clip_name, frame_id = self._parse_image_path(img_path)
        return os.path.join(self._seg_root, clip_name, frame_id + '.png')

    # def _load_landmarks(self, landmark_dir,  frame_ids, landmark_files: list[str] = None):
    #     landmarks = [np.loadtxt(self.get_landmark_path(landmark_dir, i))[:, :3] for i in frame_ids]
    #     return np.array(landmarks)

    def _load_landmarks(self, landmark_files: list[str] = None):
        landmarks = [np.loadtxt(lmfile)[:, :3] for lmfile in landmark_files]
        return np.array(landmarks)

    def _load_poses(self, pose_files: list[str]) -> np.ndarray:
        poses = []
        for path in pose_files:
            pose = np.loadtxt(path)
            poses.append(pose)
        return np.array(poses)
