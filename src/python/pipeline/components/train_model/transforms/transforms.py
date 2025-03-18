from torch import nn, float32
import torchvision.transforms.v2 as v2

class ShuffleNet_V2_X0_5_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a ShuffleNetV2_X0_5 model.
    """
    def __init__(self):
        super().__init__()

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x



class EfficientNet_B0_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a EfficientNet_B0 model.
    """
    def __init__(self):
        super().__init__()

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x