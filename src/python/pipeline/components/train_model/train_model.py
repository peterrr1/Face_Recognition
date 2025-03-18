import argparse
import mlflow
from datasets.CelebA import CelebA
from transforms.transforms import ShuffleNet_V2_X0_5_FaceTransforms
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torch import nn
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from torch.optim import AdamW



def parse_args():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training the model.")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for training the model.")



def main(args):
    print("Training model...")

    input_data = args.input_data

    ## Load the model
    model = shufflenet_v2_x0_5(weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    classifier = v2.Compose(
        nn.Dropout(p=0.2),
        nn.Linear(1024, 40)
    )
    model.fc = classifier

    ## Load the dataset
    dataset = CelebA(input_data, transform=ShuffleNet_V2_X0_5_FaceTransforms())





if __name__ == '__main__':
    args = parse_args()
    main(args)