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
        # Detect faces on the image
        preds = self.detector.predict(x, verbose=False)
        
        # If no faces are detected, return the original image
        if preds[0].boxes.conf.numel() == 0:
            return x

        # Select the one with the highest confidence if exists
        idx = argmax(preds[0].boxes.conf)

        # Get the bounding box coordinates and crop the image
        x1, y1, x2, y2 = preds[0].boxes.xyxy[idx].tolist()
        cropped_img = x.crop((x1 - self.pad, y1 - self.pad, x2 + self.pad, y2 + self.pad))
        return cropped_img




class FaceTransforms(nn.Module):
    def __init__(
            self,
            detector = None,
            pad: int = 0,
            size: tuple = (224, 224),
            interpolation_mode = v2.InterpolationMode.BILINEAR
    ):
        super().__init__()

        self.transforms = v2.Compose([
            DetectFace(detector, pad),
            v2.Resize(size, interpolation=interpolation_mode),
        ])
    
    def forward(self, x):
        x = self.transforms(x)
        return x