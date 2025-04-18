from torch import nn, float32
import torchvision.transforms.v2 as v2


class ShuffleNet_V2_X0_5_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a ShuffleNetV2_X0_5 model.
    """
    def __init__(
            self,
            inference: bool = False,
            flip_prob: int = 0.5,
            rotation_degree: int = 15
    ):
        super().__init__()


        if not inference:
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=flip_prob),
                v2.RandomRotation(degrees=rotation_degree),
                v2.ToImage(),
                v2.ToDtype(dtype=float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(dtype=float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
            ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x




class MobileNet_V2_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a ShuffleNetV2_X0_5 model.
    """
    def __init__(self, detector = None, pad: int = 0, inference: bool = False):
        super().__init__()

        self.detector = detector

        if not inference:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(dtype=float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = v2.Compose([
                v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
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
    def __init__(self, detector = None, pad: int = 0):
        super().__init__()

        self.detector = detector

        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x