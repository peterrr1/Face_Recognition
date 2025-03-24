
def get_model(model_name: str, pretrained: bool = True):
    if model_name == 'shufflenet':
        from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        return shufflenet_v2_x0_5(weights=weights)
