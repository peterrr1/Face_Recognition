import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.models import shufflenet_v2_x0_5
import os

def parse_args():
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the test data.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for testing the model.")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the test results.")

    args = parser.parse_args()
    return args

def main(args):
    model_weights_path = args.model_path
    test_data_path = args.test_data
    batch_size = args.batch_size
    output_model_path = args.output_model


    model_dict = torch.load(os.path.join(model_weights_path, 'shufflenet.pth'))
    model_name = model_dict['model_name']
    print(f"Model name: {model_name}")


    try:
        model_dict.pop('model_name')
    except KeyError:
        print("Model name not found in the model dictionary!")


    print("Loading model for testing...")
    model = shufflenet_v2_x0_5()
    model.load_state_dict(model_dict)
    print("Model loaded successfully!")

    test_set = torch.load(os.path.join(test_data_path, 'test_data.pth'))
    print(f"Test data contains: {len(test_set)} samples")

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    print(f"Test loader contains: {len(test_loader)} batches")

    print("Testing the model...")
    ## TODO

    
    model_state_dict = model.state_dict()

    ## Add the model name to the state dictionary
    model_state_dict['model_name'] = model_name

    ## Save the model to the output path
    print(f'Saving model to output path: {output_model_path}')
    torch.save(model_state_dict, os.path.join(output_model_path, f'{model_name}.pth'))
    print('Model saved successfully!')



if __name__ == '__main__':
    args = parse_args()
    main(args)
    print("Test model step completed successfully!")