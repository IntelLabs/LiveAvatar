import torch
from models.recog.model import Backbone


class FaceRec():
    def __init__(self, model_file="./assets/model_ir_se50.pth", device='cuda'):
        self.model = Backbone(50, drop_ratio=0.6, mode='ir_se').to(device)
        self.model.load_state_dict(torch.load(model_file))
        self.model.eval()
        self.input_size = (112, 112)

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        image_resized = torch.nn.functional.interpolate(image, size=self.input_size, mode="bilinear")
        return self.model(image_resized)