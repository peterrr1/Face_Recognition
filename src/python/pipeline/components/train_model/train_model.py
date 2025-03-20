import argparse
import mlflow
from torch import nn, load
import os
from torchvision.models import shufflenet_v2_x0_5



def parse_args():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights.")
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training the model.")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for training the model.")
    
    args = parser.parse_args()
    return args


def main(args):
    print("Training model...")

    ## Load the model
    model_weights_path = args.model_path
    input_data_path = args.input_data
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    ## Test prints
    print(f"Model weights path contains: {os.listdir(model_weights_path)}")
    print(f"Input data path contains: {os.listdir(input_data_path)}")
    print(f"Dataset files: {os.listdir(os.path.join(input_data_path, 'celeba', 'transformed_images'))}")
    
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")


    model_dict = load(os.path.join(model_weights_path, 'shufflenet.pth'))
    print(f"Model name: {model_dict['model_name']}")

    try:
        model_dict.pop('model_name')
    except KeyError:
        print("Model name not found in the model dictionary!")

    model = shufflenet_v2_x0_5()
    model.load_state_dict(model_dict)
    print("Model loaded successfully!")
    print(model.eval())

    


if __name__ == '__main__':
    args = parse_args()
    main(args)
    print('Training step completed successfully!')