from torch import nn

def get_model(model_name: str):
    if model_name == 'shufflenet':
        from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
        model = shufflenet_v2_x0_5()
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024, 40)
        )

        return model