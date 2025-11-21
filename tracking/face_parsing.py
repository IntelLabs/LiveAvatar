import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.utils.data as td
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import kornia

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "face_parsing_pytorch"))
from libs.face_parsing_pytorch.model import BiSeNet


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

class ImageDataset(td.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with Image.open(self.files[idx]) as img:
            img.load()
        img = img.resize((512, 512))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.files[idx]


def seg_filename(image_path):
    image_filename = os.path.split(image_path)[1]
    return os.path.splitext(image_filename)[0] + '.png'


def get_parsing_maps(parsing_anno):
    # Colors for all 20 parts

    part_colors = [[0, 0, 0],
                   [250, 0, 0],  # skin face
                   [240, 0, 0],  # l_brow
                   [230, 0, 0],  # r_brow
                   [210, 0, 0],  # l_eye
                   [200, 0, 0],  # r_eye
                   [190, 0, 0],  # eye_glasses
                   [180, 0, 0],  # l_ear
                   [170, 0, 0],  # r_ear
                   [160, 0, 0],  # ear_r
                   [150, 0, 0],  # nose
                   [140, 0, 0],  # mouth
                   [130, 0, 0],  # u_lip
                   [120, 0, 0],  # l_lip
                   [110, 0, 0],  # neck
                   [100, 0, 0],  # neck_l
                   [90, 0, 0],  # cloth
                   [80, 0, 0],  # hair
                   [70, 0, 0],  # hat
                   [60, 0, 0],
                   [50, 0, 0],
                   [40, 0, 0],
                   [30, 0, 0],
                   [20, 0, 0],
                   [10, 0, 0]]

    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    # vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3))

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    return vis_parsing_anno_color.astype(np.uint8)


def save_parsing_maps(parsing_anno, save_path='vis_results/parsing_map_on_im.jpg'):
    img = get_parsing_maps(parsing_anno)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class FaceParser():
    def __init__(self, checkpoint, batch_size, progress=True, device='cuda'):
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.load_state_dict(torch.load(checkpoint, weights_only=True))
        self.net = self.net.to(device).eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.batch_size = batch_size
        self.progress = progress
        self.device = device

    def evaluate(self, input_dir='./data', output_dir='./res/test_res', dsize=None, extension='.jpg'):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_files = sorted(glob.glob(os.path.join(input_dir, '*'+extension)))
        dataloader = DataLoader(ImageDataset(input_files, self.to_tensor), batch_size=self.batch_size)

        with torch.no_grad():

            bar = tqdm(total=len(input_files), disable=not self.progress, dynamic_ncols=True)

            for images, image_paths in dataloader:
                out = self.net(images.to(self.device))[0]
                if dsize is not None:
                    out = kornia.geometry.transform.resize(out, dsize, interpolation='nearest')
                parsing = torch.argmax(out, dim=1)
                parsing = parsing.cpu().numpy()
                for i in range(parsing.shape[0]):
                    seg_path = os.path.join(output_dir, seg_filename(image_paths[i]))
                    save_parsing_maps(parsing[i], save_path=seg_path)
                    bar.update()

    def predict(self, image):
        image = self.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            out = self.net(image.to(self.device))[0]
            parsing = torch.argmax(out, dim=1).cpu().numpy()
        return get_parsing_maps(parsing[0])[..., 0]