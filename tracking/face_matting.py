import torch
import numpy as np
from utils.util import to_numpy
from torchvision import transforms
from libs.RobustVideoMatting.model import MattingNetwork


class FaceMatting:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device
        self.transform = transforms.ToTensor()
        self.reset()

    def reset(self):
        self.rec = [None] * 4
        self.rec_flip = [None] * 4

    def convert_frame(
            self,
            image: np.ndarray,
            dtype: torch.dtype | None = None
    ):
        src = self.transform(image)

        with torch.no_grad():

            src = src.to(self.device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
            fgr, pha, *self.rec = self.model(src, *self.rec, 1)

            src_fl = torch.flip(src, dims=[-1])
            fgr_fl, pha_fl, *self.rec_flip = self.model(src_fl, *self.rec_flip, 1)
            pha = (pha + torch.flip(pha_fl, dims=[-1])) / 2.0

        return to_numpy(pha[0, 0])

