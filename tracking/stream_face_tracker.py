from __future__ import annotations
from typing import Literal
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from tracking.landmarks import LandmarkDetector
from tracking.video_processor import VideoProcessor
from tracking.alignment import CanonicalFaceAlignment
from visualization.vis import make_grid, show_image, add_landmarks_to_images


def get_bbox_from_landmarks(lms: np.ndarray, scale=2.0) -> np.ndarray:
    min_x, min_y = lms[:, :2].min(axis=0)
    max_x, max_y = lms[:, :2].max(axis=0)

    w, h = max_x - min_x, max_y - min_y

    cx, cy = min_x + w//2, min_y + h//2

    size = scale * max(w, h) / 2
    # size = 80

    return np.array([
        cx - size,
        cy - size - int(size*0.1),
        cx + size,
        cy + size - int(size*0.1),
    ])


def crop_image(
        image: np.ndarray,
        bbox: np.ndarray,
        output_size: tuple[int, int] | None = None,
        padval = 0
) -> np.ndarray:

    h, w, c = image.shape

    min_x, min_y, max_x, max_y = bbox.astype(int)
    # min_x, min_y, max_x, max_y = np.array([w-100, 100, w+156, 356]).astype(int)

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    crop = np.zeros((bbox_height, bbox_width, 3), dtype=image.dtype)
    crop.fill(padval)

    src_l = max(0, min_x)
    src_r = min(w, max_x)
    src_t = max(0, min_y)
    src_b = min(h, max_y)

    dst_l = max(0 - min_x, 0)
    dst_r = max(bbox_width - max(0, max_x - w), 0)
    dst_t = max(0 - min_y, 0)
    dst_b = max(bbox_height - max(0, max_y - h), 0)

    crop[dst_t: dst_b, dst_l: dst_r] = image[src_t: src_b, src_l: src_r]

    if output_size is not None:
        crop = cv2.resize(crop, dsize=output_size)

    return crop


class StreamFaceTracker(VideoProcessor):

    def __init__(
            self,
            asset_dir: str = "../assets",
            batch_size: int = 8,
            image_size: int = 512,
            with_segmentation: bool = False,
            fov=30.0,
            update_freq: int = 1,
            reset_interval_sec: float | None = None,
            gpu: bool = True,
            min_confidence=0.5,
            running_mode: Literal['image', 'video'] = 'image',
            flip_input: bool = False,
            scale = 1.0  # 1.75 for streaming_server GUI # 1.5 for paper fig
    ):

        super().__init__(batch_size=batch_size)

        self._image_size = image_size
        self._with_segmentation = with_segmentation
        self._frame_cnt = 0
        self._lms = None
        self._update_freq = update_freq
        self._reset_interval_sec = reset_interval_sec
        self._reset_timestamp = None
        if reset_interval_sec is not None and reset_interval_sec > 0:
            self._reset_timestamp = time.time() + reset_interval_sec

        canonical_obj_fpath = os.path.join(asset_dir, "face_model_with_iris.obj")
        focal_length = 1.0 / np.tan(np.radians(fov)) * scale
        self.camera_distance = focal_length
        self._canonical_face_alignment = CanonicalFaceAlignment(
            canonical_obj_fpath,
            cam_pos=np.array([[0.5, -0.5, self.camera_distance]]),
            look_at=np.array([[0.5, -0.5, 0.0]]),
            # cam_pos=np.array([[0.0, -10.0, focal_length]]),
            # look_at=np.array([[-0., -0., 0.0]]),
            fov=fov
        )

        self._landmark_detector = LandmarkDetector(
            asset_dir,
            running_mode=running_mode,
            flip_input=flip_input,
            gpu=gpu,
            min_confidence=min_confidence,
        )

        lib_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'libs')

        self._matting = None
        self._face_parser = None

        if with_segmentation:
            from tracking.face_matting import FaceMatting
            self._matting = FaceMatting(
                variant='resnet50',
                checkpoint=os.path.join(lib_dir, 'RobustVideoMatting/rvm_resnet50.pth'),
                device=self._device,
            )
            from tracking.face_parsing import FaceParser
            self._face_parser = FaceParser(
                checkpoint=os.path.join(lib_dir, 'face_parsing_pytorch/res/cp/79999_iter.pth'),
                batch_size=batch_size,
                progress=self._progress,
                device=self._device
            )


    def track(self, frame, frame_timestamp_ms, crop_input=True, show=False) -> dict | None:

        # frame = cv2.resize(frame, dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)

        self._frame_cnt += 1

        self._landmark_detector.reset()
        if  self._reset_timestamp is not None and time.time()  > self._reset_timestamp:
            print("Resetting face tracker...")
            self._landmark_detector.reset()
            self._lms = None
            self._reset_timestamp = time.time() + self._reset_interval_sec

        if self._lms is None or self._frame_cnt % self._update_freq == 0:
            self._lms = self._landmark_detector.detect(frame, frame_timestamp_ms, show=show)
            if self._lms is None:
                return None
            self._lms[:, 0] *= frame.shape[1]
            self._lms[:, 1] *= frame.shape[0]

        lms = self._lms.copy()

        if crop_input:
            bbox = get_bbox_from_landmarks(lms, scale=2.0)
            crop = crop_image(frame, bbox, output_size=(self._image_size, self._image_size))
            crop_masked = crop.copy()
        else:
            bbox = np.array([0, 0, frame.shape[1], frame.shape[0]])
            crop = frame.copy()
            crop_masked = frame.copy()

        results = {}

        if self._with_segmentation:
            alpha = self._matting.convert_frame(crop)
            seg = self._face_parser.predict(crop)

            results |= dict(alpha=alpha, seg=seg)

            mask = alpha > 0.7
            # mask[seg == 0] = 0
            mask[seg == 90] = 0
            crop_masked[~mask.astype(bool), :] = 0

            grid = make_grid([crop, alpha, seg, crop_masked])
        else:
            grid = make_grid([crop_masked])

        # convert full frame landmarks to crop based landmarks
        lms_crop = lms.copy()
        lms_crop[:, 0] -= bbox[0]
        lms_crop[:, 1] -= bbox[1]
        lms_crop[:, 0] /= bbox[2] - bbox[0]
        lms_crop[:, 1] /= bbox[3] - bbox[1]

        # lms_crop[:, 0] *= self._image_size
        # lms_crop[:, 1] *= self._image_size
        # crop_with_lms = add_landmarks_to_images([crop], lms_crop[np.newaxis])[0]
        # show_image("landmark", crop_with_lms, wait=0)
        # lms_crop2 = self._landmark_detector.detect(crop, show=True)

        # NOTE: coordinate frame is different between 2D image x/y and 3D splatting x/y
        _lms = lms_crop.copy()
        _lms[:, 1] *= -1
        _lms[:, 2] *= -1
        cam, lms_new, M_inv = self._canonical_face_alignment.align(_lms)

        # if show:
        #     show_image("tracking results", grid, wait=1)

        results |= dict(
            crop=crop_masked,
            keypoints=lms,
            keypoints_crop=lms_crop,
            camera=cam,
            keypoints_aligned=lms_new,
        )

        return results
