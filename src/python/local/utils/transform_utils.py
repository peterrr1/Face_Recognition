from torch import nn, argmax, float32
import torchvision.transforms.v2 as v2


class DetectFace(nn.Module):
    """
    Detects a face in an image using a face detector and crops the image around the face.
    """
    def __init__(self, detector, pad: int = 0):
        super().__init__()
        self.detector = detector
        self.pad = pad

    def forward(self, x):
        if self.detector is None:
            return x
        
        x = self._forward(x)
        return x

    def _forward(self, x):
        ## Detect faces on the image
        preds = self.detector.predict(x, verbose=False)
        
        ## If no faces are detected, return the original image
        if preds[0].boxes.conf.numel() == 0:
            return x

        # Select the one with the highest confidence if exists
        idx = argmax(preds[0].boxes.conf)

        # Get the bounding box coordinates and crop the image
        x1, y1, x2, y2 = preds[0].boxes.xyxy[idx].tolist()
        cropped_img = x.crop((x1 - self.pad, y1 - self.pad, x2 + self.pad, y2 + self.pad))
        return cropped_img


class ShuffleNet_V2_X0_5_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a ShuffleNetV2_X0_5 model.
    """
    def __init__(self, detector = None, pad: int = 0, inference: bool = False):
        super().__init__()

        self.detector = detector

        if not inference:
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                DetectFace(self.detector, pad),
                v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
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
    


class MobileNet_V2_FaceTransforms(nn.Module):
    """
    A series of transformations to apply to an image before feeding it to a ShuffleNetV2_X0_5 model.
    """
    def __init__(self, detector = None, pad: int = 0, inference: bool = False):
        super().__init__()

        self.detector = detector

        if not inference:
            self.transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=15),
                DetectFace(self.detector, pad),
                v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
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
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            DetectFace(detector, pad),
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(dtype=float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x