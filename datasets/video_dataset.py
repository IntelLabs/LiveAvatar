import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

import pandas as pd
import torch.utils.data as td
import glob
import json

from configs.config import EXPRESSION_KP_IDS, KP_IDS_IRISES
from training.loss_utils import create_face_weights


FACE_SURFACE_PARTS = [
    250,  # skin face
    240,  # l_brow
    230,  # r_brow
    210,  # l_eye
    200,  # r_eye
    # 190,  # eye_glasses
    # 180,  # l_ear
    # 170,  # r_ear
    # 160,  # ear_r
    150,  # nose
    140,  # mouth
    130,  # u_lip
    120,  # l_lip
]
# from tracking.face_tracker import find_face_masks

from tracking.landmarks import flip_landmarks


def flip_pose_np(R: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert len(R.shape) == 2
    assert len(T.shape) == 1
    R_ = R.copy()
    T_ = T.copy()
    R_[0, 1:3] *= -1
    R_[1:3, 0] *= -1
    T_[0] *= -1
    return R_, T_


# from concave_hull import concave_hull
def find_minimal_contour_mask(points: np.array, h: int, w: int):
    mask = np.zeros((h, w), dtype=np.uint8)
    contour = concave_hull(points, concavity=2)
    if contour is not None:
        cv2.fillPoly(mask, [contour], 255)
    return mask


class VideoDataset(td.Dataset):
    def __init__(
            self,
            root: str,
            data_folder: str,
            transform = None,
            pixelwise_transform = None,
            corruption_transform = None,
            max_clip_length: int = 200,
            num_frames: int = 16,
            st: int = 0,
            nd: int | None = None,
            train: bool = True,
            blacklist_clips: list[str] | None = None,
            blacklist_videos: list[str] | None = None,
            mask_inputs: bool = True,
            cloth: bool = False,
            filter_outliers: bool = True,
            return_face_weights: bool = False,
            background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
            filter: dict | None = None,
            reenact: bool = False,
            cross_video: bool = False,
            n_images_per_clip: int = 2,
            p_flip_train: float = 0.3
    ):
        assert os.path.exists(root), f"{self.__class__.__name__} root path '{root}' does not exist!"
        assert os.path.exists(os.path.join(root, data_folder)), \
            f"{self.__class__.__name__} data folder '{data_folder}' does not exist at location '{root}'!"

        self.transform = transform
        self.pixelwise_transform = pixelwise_transform
        self.corruption_transform = corruption_transform
        self.return_face_weights = return_face_weights
        self.reenact = reenact
        self.cross_video = cross_video
        self.n_images_per_clip = n_images_per_clip

        if filter is None:
            filter = {}
        self.filter = filter

        asset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../assets")
        self._mirror_ids = np.loadtxt( os.path.join(asset_dir, "mediapipe_mirror_ids_obj.txt")).astype(int)
        self.p_flip = p_flip_train if train else 0

        self.train = train
        self.mask_inputs = mask_inputs
        self.cloth = cloth
        self.background_color = background_color

        self.image_folder = 'frames'
        self.frame_folder = 'frames'
        self.matted_folder = 'matted'
        self.alpha_folder = 'alpha'
        self.pose_folder = 'poses'
        self.seg_folder = 'seg_mask'
        self.metadata_folder = 'metadata'

        self.frame_meta_filename = 'frame_meta.csv'

        self.image_dir = os.path.join(root, data_folder, self.image_folder)
        self.pose_dir = os.path.join(root, data_folder, self.pose_folder)
        self.metadata_dir = os.path.join(root, data_folder, self.metadata_folder)

        def clip_name(path):
            return os.path.split(path)[1]


        def frame_id(path):
            return int(os.path.splitext(os.path.split(path)[1])[0].split('_')[0])

        def list_frames(path):
            frames = sorted(glob.glob(os.path.join(path, '*_pose.txt')))[:max_clip_length]
            ids = np.linspace(0, len(frames)-1, num=min(num_frames, len(frames))).astype(int)
            selected_frames = [frames[i] for i in ids]
            return selected_frames

        clip_names = self.load_clip_names(root, st, nd)

        # remove blacklisted clips
        if blacklist_clips is not None:
            clip_names = [clip for clip in clip_names if clip not in set(blacklist_clips)]

        if blacklist_videos is not None:
            _video_names = [self.vid_name(clip) for clip in clip_names]
            clip_names = [clip for (clip, vid) in zip(clip_names, _video_names) if vid not in set(blacklist_videos)]

        clip_paths = [os.path.join(self.pose_dir, clip) for clip in clip_names]

        clips_to_frames = {clip_name(p): list_frames(p) for p in clip_paths}

        list_frames = []
        list_vid_names = []
        list_clip_ids = []
        list_clip_names = []

        for clip_id, (clip_name, frame_paths) in enumerate(clips_to_frames.items()):
            n_frames = len(frame_paths)
            list_frames.extend(frame_paths)
            list_clip_ids.extend([clip_id + st] * n_frames)
            list_vid_names.extend([self.vid_name(clip_name)] * n_frames)
            list_clip_names.extend([clip_name] * n_frames)

        unique_vids = np.unique(list_vid_names)
        vid_name_to_id = {vid: idx for vid, idx in zip(np.unique(unique_vids), range(len(unique_vids)))}
        list_vid_ids = [vid_name_to_id[vid] for vid in list_vid_names]

        list_frame_ids = [frame_id(path) for path in list_frames]

        self.meta_data = pd.DataFrame({
            'vid_name': list_vid_names,
            'vid_id': list_vid_ids,
            'clip_name': list_clip_names,
            'frame_id': list_frame_ids,
            'pose_path': list_frames
        })

        df = self.load_metadata(clip_names)

        self.meta_data = pd.merge(self.meta_data, df, on=['clip_name', 'frame_id'])

        if filter_outliers and train:
            # filter out low quality frames
            self.meta_data = self.meta_data.query("camera_distance < 3.0 & luminance > 30 & face_area > 0.04")
            self.meta_data = self.meta_data.query("reproj_error < 10.0")
            self.meta_data = self.meta_data.query("nonzero_count > 50000")

        # select data subset
        if self.train or True:
            self.meta_data = self.meta_data.query(f"azimuth_max_abs > 45 or azimuth_std > {filter.get('min_azimuth_std', 0)}")
            self.meta_data = self.meta_data.query(f"azimuth_range > {filter.get('min_azimuth_range', 0)}")
            # self.meta_data = self.meta_data.query(f"azimuth_std > {filter.get('min_azimuth_std', 0)}")

        self.meta_data = self.meta_data.reset_index(drop=True)
        self.meta_data['iloc'] = self.meta_data.index

        self.clip_azimuth_std = self.meta_data.groupby('clip_id').azimuth_std.mean()
        self.vid_azimuth_std = self.meta_data.groupby('vid_id').azimuth_std.mean()

    def vid_name(self, clip_name):
        return "".join(clip_name.split('_')[:-1])

    def load_clip_names(self, root: str, st: int, nt: int):
        raise NotImplementedError()

    def load_metadata(self, clip_names) -> pd.DataFrame:
        list_frame_meta = []
        for cname in clip_names:
            metafile = os.path.join(self.metadata_dir, cname, self.frame_meta_filename)
            if os.path.isfile(metafile):
                list_frame_meta.append(pd.read_csv(metafile))

        df = pd.concat(list_frame_meta)

        # add some stats
        df["azimuth"] = np.degrees(np.arctan(df.camera_x / df.camera_z))
        df["elevation"] = np.degrees(np.arctan(df.camera_y / df.camera_distance))
        df['azimuth_abs'] = df.azimuth.abs()
        df['azimuth_max'] = df.groupby('clip_id').azimuth.transform('max')
        df['azimuth_min'] = df.groupby('clip_id').azimuth.transform('min')
        df['azimuth_range'] = df['azimuth_max'] - df['azimuth_min']
        df['azimuth_std'] = df.groupby('clip_id').azimuth.transform('std')
        df['azimuth_max_abs'] = df.groupby('clip_id').azimuth_abs.transform('max')
        return df

    def __len__(self):
        return len(self.meta_data)

    def __str__(self):
        s =  f"Dataset {self.__class__.__name__}  (train={self.train})\n"
        s += f"-" * 50
        s += f"\n\tNumber of videos: {len(self.get_video_names())}"
        s += f"\n\tNumber of clips : {len(self.get_clip_names())}"
        s += f"\n\tNumber of frames: {len(self)}"
        return s

    def get_video_id_from_path(self, path):
        raise NotImplementedError

    def get_clip_names(self):
        return self.meta_data.clip_name.unique()

    def get_clip_ids(self):
        return self.meta_data.clip_id.unique()

    @property
    def clip_ids(self):
        return list(self.meta_data.clip_id)

    def get_video_names(self):
        return self.meta_data.vid_name.unique()

    def _select_view(self, anchor: dict | list[dict], candidates: pd.DataFrame, mode="max_dist", n=1):
        if mode == "max_dist":
            distant_view_id = (candidates.azimuth - anchor['azimuth']).abs().argmax()
            return candidates.iloc[distant_view_id].name  # NOTE: assumes a sequential index
        elif mode == "dist_sample":
            if isinstance(anchor, dict):
                anchor = [anchor]
            dists = []
            for i in range(len(anchor)):
                dists.append((candidates.azimuth - anchor[i]['azimuth']).abs().values)
            dists = np.array(dists).min(axis=0)
            try:
                return candidates.sample(weights=dists**2, n=n).iloc[0].name
            except ValueError:
                return candidates.sample(n=1).iloc[0].name
        else:
            raise NotImplementedError()
            # candidate_ilocs = [k for k in np.where(is_same_clip)[0] if k != idx]
            # source_idx = np.random.choice(candidate_ilocs, 1)[0] if len(candidate_ilocs) > 0 else idx

    def __getitem__(self, idx):

        flip = False

        if self.train:
            # select video by azimuth variance
            clip_id = self.clip_azimuth_std.sample(weights=self.clip_azimuth_std, n=1).index.values[0]
            # select random frame in video
            idx = self.meta_data[self.meta_data.clip_id == clip_id].sample(n=1).iloc[0].name
            flip = random.random() < self.p_flip

        samples = [self.get_sample(idx, flip=flip)]

        if self.n_images_per_clip == 1:
            return samples
        else:
            # is_same_clip = self.meta_data.clip_id == samples[0]['clip_id']
            is_same_clip = self.meta_data.vid_id == samples[0]['vid_id']
            if self.reenact:
                is_first_frame = self.meta_data.frame_id == 0
                source_idx = np.where(is_same_clip & is_first_frame)[0][0]
                samples.insert(0, self.get_sample(source_idx))
            else:
                for i in range(self.n_images_per_clip-1):
                    source_idx = self._select_view(
                        anchor=samples,
                        candidates=self.meta_data[is_same_clip],
                        mode="dist_sample"
                    )
                    _flip = flip
                    # _flip = random.random() < self.p_flip
                    # if random.random() < 0.20:
                    #     _flip = not flip
                    s = self.get_sample(source_idx, flip=_flip)
                    samples.append(s)
            return samples

    def get_sample(self, idx: int, flip: bool = False) -> dict:

        sample = {}

        frame_info = self.meta_data.iloc[idx]

        pose_path = frame_info.pose_path
        keypoint_path = pose_path.replace('_pose.txt', '_landmarks.txt')
        image_path = pose_path.replace(self.pose_folder, self.image_folder).replace('_pose.txt', '.jpg')
        matted_path = pose_path.replace(self.pose_folder, self.matted_folder).replace('_pose.txt', '.png')
        seg_path = pose_path.replace(self.pose_folder, self.seg_folder).replace('_pose.txt', '.png')
        alpha_path = pose_path.replace(self.pose_folder, self.alpha_folder).replace('_pose.txt', '.jpg')

        keypoints = np.loadtxt(keypoint_path)[:, :3]

        keypoint_aligned_path = pose_path.replace('_pose.txt', '_landmarks_aligned.txt')
        keypoints_aligned = np.loadtxt(keypoint_aligned_path)[:, :3]

        pose = np.loadtxt(pose_path)
        R = pose[:3, :3]
        T = pose[:3, 3]

        if os.path.isfile(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(matted_path, cv2.IMREAD_UNCHANGED)

        def get_matting_alpha(image):
            if image.shape[2] == 4:
                return image[..., 3]
            else:
                img = cv2.imread(alpha_path)[..., 2]
                return cv2.resize(img, dsize=(image.shape[1], image.shape[0]))

        alpha = get_matting_alpha(image)
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)[..., 2]

        face_mask = np.zeros_like(seg)
        for part in FACE_SURFACE_PARTS:
            face_mask[seg == part] = 255

        # face_mask = cv2.dilate(face_mask, kernel=np.ones((5, 5), np.uint8), iterations=1)
        # plt.imshow(face_mask)
        # plt.show()
        h, w = face_mask.shape[:2]
        lms2d = keypoints.copy()
        lms2d[:, 0] *= h
        lms2d[:, 1] *= w
        lms2d = lms2d.astype(int)

        keypoints_occ = face_mask[lms2d[:, 1].clip(0, h-1), lms2d[:, 0].clip(0, w-1)] == 0

        # create foreground mask by combining matting result and face segmentation result
        # mask = alpha > 180
        mask = alpha.astype(np.float32) / 255.0
        if abs(frame_info.azimuth) < 80:
            mask[seg == 0] = 0

        # remove cloth / upper body area, if necessary
        if not self.cloth:
            mask[seg == 90] = 0

        # add area covered by 2D landmarks to mask
        # face_seg_lms = find_minimal_contour_mask(lms2d[:, :2], h, w)
        # mask[face_seg_lms > 0] = 1
        # plt.imshow(face_seg_lms)
        # plt.show()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask = mask.astype(np.uint8)

        if flip:
            R, T = flip_pose_np(R, T)
            image = image[:, ::-1]
            mask = mask[:, ::-1]
            keypoints = flip_landmarks(keypoints, mirror_ids=self._mirror_ids)
            keypoints_aligned[:, 0] *= -1

        if self.pixelwise_transform is not None:
            image = self.pixelwise_transform(image=image)['image']
            image[mask == 0] = 0  # don't transform background

        image = image.astype(np.float32) / 255.0

        input = image.copy()

        if self.corruption_transform is not None:
            input = self.corruption_transform(image=input)['image']

        # if random.random() < self.p_flip:
        #     input = input[:, ::-1]

        transformed = self.transform(image=image, mask=mask)
        target = transformed['image']
        mask = transformed['mask']
        transformed = self.transform(image=input)
        # transformed = self.transform(image=input, mask=face_mask)
        input = transformed['image']
        # face_mask = transformed['mask']

        # apply foreground mask to target image and optionally to input image
        for i in range(3):
            # target[i, ~mask.bool()] = self.background_color[i]
            target[i] = target[i] * mask + (1.0-mask) * self.background_color[i]
            if self.mask_inputs:
                input[i] = input[i] * mask + (1.0 - mask) * self.background_color[i]
                # input[i, ~mask.bool()] = self.background_color[i]

        sample.update({
            'fid': frame_info.frame_id,
            'vid_id': frame_info.vid_id,
            'vid_name': frame_info.vid_name,
            'clip_id': frame_info.clip_id,
            'clip_name': frame_info.clip_name,
            'R': R,
            'T': T,
            'input': input.clip(min=0, max=1),
            'target': target,
            'mask': mask.bool(),
            # 'face_mask': face_mask.bool(),
            'keypoints': keypoints,                 # location in normalized image space [0,1]
            'keypoints_aligned': keypoints_aligned, # location in 3D space
            'keypoints_occ': keypoints_occ,         # landmark occluded by hair, glasses, etc. (bool)
            'azimuth': frame_info.azimuth,
            'azimuth_std': frame_info.azimuth_std,
            'elevation': frame_info.elevation,
            'nonzero_count': frame_info.nonzero_count,
        })

        if self.return_face_weights:
            keypoints_2d_abs = keypoints[:, :2] * target.shape[-2:][::-1]
            sample['face_weights'] = create_face_weights(target,
                                                         keypoints_2d_abs,
                                                         keypoint_ids=EXPRESSION_KP_IDS,
                                                         keypoint_ids_iris=KP_IDS_IRISES,
                                                         w_iris=1.0,
                                                         radius=9)
        return sample

