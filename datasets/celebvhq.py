import os
import pandas as pd
import json

from datasets.video_dataset import VideoDataset


class CelebVHQ(VideoDataset):

    def __init__(
            self,
            root: str,
            data_folder: str = 'processed_celebvhq',
            st: int = 0,
            nd: int | None = None,
            **kwargs
    ):
        super().__init__(root, data_folder=data_folder, st=st, nd=nd, **kwargs)

    def load_clip_names(self, root, st, nd):
        json_path = os.path.join(root, 'celebvhq_info.json')
        with open(json_path) as f:
            data_dict = json.load(f)

        clip_names = list(data_dict['clips'].keys())
        clip_names = clip_names[st:nd]

        df_attributes = self.get_attributes_from_json(data_dict, clip_names)

        # filter clips by attribute (action or appearance)
        clip_names = list(df_attributes.query("wearing_mask == 0 and kiss == 0").clip_name)

        return clip_names

    def get_attributes_from_json(self, data_dict, clip_names):
        clip_attributes = []
        action_mapping = data_dict['meta_info']['action_mapping']
        appearance_mapping = data_dict['meta_info']['appearance_mapping']
        act_to_id = {action: idx for idx, action in zip(range(len(action_mapping)), action_mapping)}
        app_to_id = {appearance: idx for idx, appearance in zip(range(len(appearance_mapping)), appearance_mapping)}

        for cl_name in clip_names:
            action = data_dict['clips'][cl_name]['attributes']['action']
            appearance = data_dict['clips'][cl_name]['attributes']['appearance']
            clip_attributes.append(
                {
                    'clip_name': cl_name,
                    'kiss': action[act_to_id['kiss']],
                    'eat': action[act_to_id['eat']],
                    'drink': action[act_to_id['drink']],
                    'turn': action[act_to_id['turn']],
                    'wearing_mask': appearance[app_to_id['wearing_mask']],
                }
            )
        return pd.DataFrame(clip_attributes)


def load_data(file_path, st, nd):
    with open(file_path) as f:
        data_dict = json.load(f)

    clip_ids = list(data_dict['clips'].keys())

    if st is None or st < 0:
        st = 0

    if nd is None:
        nd = len(clip_ids)

    for idx, key in zip(range(st, nd), clip_ids[st:nd]):
        val = data_dict['clips'][key]
        save_name = key+".mp4"
        ytb_id = val['ytb_id']
        time = val['duration']['start_sec'], val['duration']['end_sec']

        bbox = [val['bbox']['top'], val['bbox']['bottom'],
                val['bbox']['left'], val['bbox']['right']]
        yield idx, ytb_id, save_name, time, bbox


if __name__ == '__main__':

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import albumentations as alb
    from albumentations.pytorch import transforms as alb_torch
    import numpy as np
    import cv2
    import torch.utils.data as td
    from pytorch3d.renderer import FoVPerspectiveCameras

    from utils.nn import to_numpy
    from utils.util import backproject_landmarks
    from visualization.vis import make_grid
    from visualization import vis

    root = "../../../data/datasets/CelebV-HQ"

    data_root = os.path.join(root, 'processed_celebvhq')

    transform = alb.Compose([
        alb.Resize(height=256, width=256),
        alb_torch.ToTensorV2()
    ])
    # corruption_transform = alb.Compose([
    #     alb.GridDropout(unit_size_min=50, unit_size_max=150, p=1),
    # ])

    ds = CelebVHQ(
        root=root,
        st=0,
        nd=None,
        num_frames=16,
        max_clip_length=1000,
        transform=transform,
        train=False,
        # corruption_transform=corruption_transform
        filter=dict(min_azimuth_range=60),
        n_images_per_clip=3
    )
    print(ds)

    # dataloader = td.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    # from datasets.sampler import MPerClassSampler
    # sampler = MPerClassSampler(ds.clip_ids, 16, 6, len(ds), shuffle=False)
    # dataloader = td.DataLoader(ds, batch_size=64, num_workers=0, sampler=sampler)
    dataloader = td.DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)

    for data, data2, data3 in dataloader:
        print(data['clip_name'], data['clip_id'])
        print(data['azimuth'])
        print(data['elevation'])
        print(data['nonzero_count'])

        target_images = data['target']
        H, W = target_images.shape[-2:]

        # face_masks = data['face_mask']

        # keypoints = to_numpy(data['keypoints'][..., :2]) * np.array([H, W])

        cameras = FoVPerspectiveCameras(R=data['R'], T=data['T'], fov=30)
        keypoints_aligned = to_numpy(data['keypoints_aligned'][..., :3])
        keypoints = [backproject_landmarks(keypoints_aligned[i], cameras[i], W, H) for i in range(len(cameras))]

        occ = to_numpy(data['keypoints_occ'])
        keypoints_occ = keypoints * occ[..., np.newaxis]
        keypoints_vis = keypoints * ~occ[..., np.newaxis]

        target_images = vis.add_label_to_images(target_images, data['clip_name'], size=0.5)
        target_images = vis.add_label_to_images(target_images, data['clip_id'], size=0.5, loc='tl+1')
        target_images = vis.add_label_to_images(target_images, data['fid'], size=0.5, loc='tl+2')
        target_images = vis.add_error_to_images(target_images, data['azimuth'], loc='br', # cl=(255, 255, 255),
                                                precision=1, vmin=-90, vmax=90, cmap="RdBu")
        target_images = vis.add_label_to_images(
            target_images, [f"{val :.2f}" for val in data['azimuth_std']], size=0.5, loc='tr', suffix=""
        )
        target_images = vis.add_landmarks_to_images(target_images, landmarks=keypoints_occ, color=(255, 0, 0))
        target_images = vis.add_landmarks_to_images(target_images, landmarks=keypoints_vis, color=(0, 255, 0))

        grid_images = make_grid(target_images, ncols=8)

        other_images1 = vis.add_error_to_images(data2['target'], data2['azimuth'], loc='br', precision=1, vmin=-90, vmax=90, cmap="RdBu")
        other_images2 = vis.add_error_to_images(data3['target'], data3['azimuth'], loc='br', precision=1, vmin=-90, vmax=90, cmap="RdBu")

        # face_masks = vis.add_landmarks_to_images(face_masks, landmarks=keypoints, color=(255, 255, 255))
        # grid_fmasks = make_grid(face_masks, ncols=8)

        disp_data = make_grid([
            make_grid(target_images, ncols=16),
            make_grid(other_images1, ncols=16),
            make_grid(other_images2, ncols=16),
        ], ncols=1)

        cv2.imshow("CelebHQ", cv2.cvtColor(disp_data, cv2.COLOR_RGB2BGR))

        # img = to_numpy(data['target'].permute(0, 2, 3, 1))[0]
        # mask = to_numpy(data['mask'][0])
        # img[~mask] = 0.5
        # cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # img = to_numpy(data['corrupted'].permute(0, 2, 3, 1))[0]
        # cv2.imshow("corrupted", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        # plt.imshow(mask)
        # plt.show()
