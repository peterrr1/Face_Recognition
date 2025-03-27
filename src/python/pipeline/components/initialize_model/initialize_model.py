import argparse
import torch
import os

"""
initialize_model.py

This script provides functionality to initialize and save pre-trained deep learning models 
from the torchvision library. The supported models include ShuffleNet, MobileNet, and EfficientNet. 
The script allows users to specify the model name, whether to use pre-trained weights, 
and the output path for saving the model's state dictionary.

Functions:
    parse_args():
        Parses command-line arguments for model initialization.

    init_shufflenet(pretrained=True):
        Initializes a ShuffleNet model with optional pre-trained weights.

    init_mobilenet(pretrained=True):
        Initializes a MobileNet model with optional pre-trained weights.

    init_efficientnet(pretrained=True):
        Initializes an EfficientNet model with optional pre-trained weights.

    main(args):
        Main function to initialize the specified model and save its state dictionary.

Usage:
    Run the script from the command line with the following arguments:
        --model_name: Name of the model to initialize ('shufflenet', 'mobilenet', 'efficientnet').
        --pretrained: Whether to use pre-trained weights (default: True).
        --model_path: Path to save the model's state dictionary.

Example:
    python initialize_model.py --model_name shufflenet --pretrained True --model_path ./models

"""


def parse_args():
    print("Parsing arguments...")

    parser = argparse.ArgumentParser('Initialize model')
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model.", choices=['shufflenet', 'mobilenet', 'efficientnet'])
    parser.add_argument('--pretrained', type=bool, default=True, help="Whether to use pretrained weights.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights.")

    args = parser.parse_args()
    return args



def init_shufflenet(pretrained = True):
    """
    Initialize a ShuffleNet model.
    """
    from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
    weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
    model = shufflenet_v2_x0_5(weights = weights)

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(1024, 40)
    )

    return model



def init_mobilenet(pretrained = True):
    """
    Initialize a MobileNet model.
    """
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights = weights)

    return model



def init_efficientnet(pretrained = True):
    """
    Initialize an EfficientNet model.
    """
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights = weights)
    return model




def main(args):

    ## Parse the arguments
    model_name = args.model_name
    pretrained = args.pretrained
    output_path = args.model_path

    ## Initialize the model to None
    model = None

    ## Initialize the model according to the model name
    if model_name == 'shufflenet':
        print('Initializing ShuffleNet model...')
        model = init_shufflenet(pretrained)

    elif model_name == 'mobilenet':
        print('Initializing MobileNet model...')
        model = init_mobilenet(pretrained)

    elif model_name == 'efficientnet':
        print('Initializing EfficientNet model...')
        model = init_efficientnet(pretrained)

    ## Define the model state dictionary
    model_state_dict = model.state_dict()

    ## Add the model name to the state dictionary
    model_state_dict['model_name'] = model_name


    ## Save the model to the output path
    print(f'Saving model to output path: {output_path}')
    torch.save(model_state_dict, os.path.join(output_path, f'{model_name}.pth'))
    print('Model saved successfully!')

if __name__ == '__main__':
    
    args = parse_args()
    main(args)
    print("Model initialization step completed successfully!")
