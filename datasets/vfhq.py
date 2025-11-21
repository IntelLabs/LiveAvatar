import os
from datasets.video_dataset import VideoDataset


class VFHQ(VideoDataset):

    def __init__(
            self,
            root: str,
            data_folder: str = 'processed_vfhq',
            st: int = 0,
            nd: int | None = None,
            **kwargs
    ):
        super().__init__(root, data_folder=data_folder, st=st, nd=nd, **kwargs)

    def load_clip_names(self, root, st, nd):
        return sorted(os.listdir(self.pose_dir))[st:nd]

    def vid_name(self, clip_name):
        return "+".join(clip_name.split('+')[1:3])


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

    root = "../../../data/datasets/VFHQ/train"

    transform = alb.Compose([
        alb.Resize(height=280, width=280),
        alb_torch.ToTensorV2()
    ])

    ds = VFHQ(root=root, st=34, nd=100, transform=transform, p_flip_train=0.5, train=True, cloth=True,
              background_color=(1,1,1))
    print(ds)

    dataloader = td.DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)

    for (data, data2) in dataloader:

        print(data['clip_name'], data['clip_id'])
        print(data['azimuth'])
        print(data['elevation'])
        print(data['nonzero_count'])

        target_images = data['target']
        drive_images = data2['target']
        H, W = target_images.shape[-2:]

        cameras = FoVPerspectiveCameras(R=data['R'], T=data['T'], fov=30)
        keypoints_aligned = to_numpy(data['keypoints_aligned'][..., :3])
        keypoints = [backproject_landmarks(keypoints_aligned[i], cameras[i], W, H) for i in range(len(cameras))]
        # keypoints = to_numpy(data['keypoints'][..., :2]) * np.array([H, W])

        occ = to_numpy(data['keypoints_occ'])
        keypoints_occ = keypoints * occ[..., np.newaxis]
        keypoints_vis = keypoints * ~occ[..., np.newaxis]

        target_images = vis.add_label_to_images(target_images, data['clip_name'], size=0.4)
        target_images = vis.add_label_to_images(target_images, data['clip_id'], size=0.5, loc='tl+1')
        target_images = vis.add_label_to_images(target_images, data['vid_id'], size=0.5, loc='tl+2')
        target_images = vis.add_error_to_images(target_images, data['azimuth'], loc='br', # cl=(255, 255, 255),
                                                precision=1, vmin=-90, vmax=90, cmap="RdBu")
        target_images = vis.add_label_to_images(target_images, data['azimuth_std'].int(), loc='tr', suffix="")
        target_images = vis.add_landmarks_to_images(target_images, landmarks=keypoints_occ, color=(255, 0, 0))
        target_images = vis.add_landmarks_to_images(target_images, landmarks=keypoints_vis, color=(0, 255, 0))

        drive_images = vis.add_label_to_images(drive_images, data2['clip_name'], size=0.4)
        drive_images = vis.add_label_to_images(drive_images, data2['clip_id'], size=0.5, loc='tl+1')
        drive_images = vis.add_label_to_images(drive_images, data2['vid_id'], size=0.5, loc='tl+2')

        disp_data = make_grid([
            make_grid(target_images, ncols=16),
            make_grid(drive_images, ncols=16),
        ], ncols=1)

        cv2.imshow("VFHQ", cv2.cvtColor(disp_data, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
