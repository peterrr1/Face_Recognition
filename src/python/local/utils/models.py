from torchvision.models import mobilenet_v2, efficientnet_b0, shufflenet_v2_x0_5
from torchvision.models import MobileNet_V2_Weights, ShuffleNet_V2_X0_5_Weights, EfficientNet_B0_Weights
from torch import nn



class ShuffleNetV2_X0_5(nn.Module):
    def __init__(
            self,
            in_features: int = 1024,
            out_features: int = 40,
            pretrained: bool = True,
            weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1,
            freeze: bool = True
    ):
        super().__init__()


        classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=out_features),
        )

        self.model = shufflenet_v2_x0_5(weights=weights) if pretrained else shufflenet_v2_x0_5()

        self.model.fc = classifier


        if freeze:
            self._freeze_layers()
    


    def forward(self, x):
        x = self.model(x)
        return x


    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True





class MobileNetV2(nn.Module):
    def __init__(
            self,
            in_features: int = 1280,
            out_features: int = 40,
            pretrained: bool = True,
            weights=MobileNet_V2_Weights.IMAGENET1K_V2,
            freeze: bool = True
    ):
        super().__init__()

        classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=out_features),
        )

        self.model = mobilenet_v2(weights=weights) if pretrained else mobilenet_v2()

        self.model.classifier = classifier

        if freeze:
            self._freeze_layers()


    def forward(self, x):
        x = self.model(x)
        return x
    

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    



class EfficientNetB0(nn.Module):
    def __init__(
            self,
            in_features: int = 1280,
            out_features: int = 40,
            pretrained: bool = True,
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1,
            freeze: bool = True
    ):
        super().__init__()


        classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=out_features),
        )

        self.model = efficientnet_b0(weights=weights) if pretrained else efficientnet_b0()

        self.model.classifier = classifier

        if freeze:
            self._freeze_layers()


    def forward(self, x):
        x = self.model(x)
        return x
    

    def _freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
