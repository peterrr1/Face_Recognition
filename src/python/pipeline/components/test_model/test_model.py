import argparse
import torch
from torch.utils.data import DataLoader
from utils.datasets.CelebA import CelebA
from utils.models import get_model
from utils.transforms import ShuffleNet_V2_X0_5_FaceTransforms
from utils.metrics import evaluate_performance, create_zero_metrics, MetricsLogger
from utils.common import divide_dict, add_dicts
from utils.constants import celeba_columns
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
import mlflow
import os
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the test data.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for testing the model.")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the test results.")

    args = parser.parse_args()
    return args


def test(model, loader, criterion, logger, device):
    with torch.no_grad():
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()

        for batch, (input, target) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)

            pred = model(input)
            loss = criterion(pred, target)

            metrics = evaluate_performance(target.detach(), pred.detach(), threshold=0.5)
            loss_sum += loss.item()
            metrics_sum = add_dicts(metrics_sum, metrics)

        ## Calculate the average loss and metrics
        avg_loss = loss_sum / len(loader)

        try:
            avg_metrics = divide_dict(metrics_sum, len(loader))
        except ZeroDivisionError as e:
            print('ZeroDivisionError: ', e)
            avg_metrics = metrics_sum 

        ## Add the loss to the metrics
        avg_metrics['loss'] = avg_loss

        ## Log the metrics
        logger.log_metrics('TEST', avg_metrics, 0)
        logger.save_artifact()

        print(f'TEST - Loss: {avg_loss}')



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

    model = get_model(model_name)
    model.load_state_dict(model_dict)

    print("Model loaded successfully!")


    ## Load the test data
    test_ds = CelebA(test_data_path, transform=ShuffleNet_V2_X0_5_FaceTransforms())


    ## For testing purposes only, create daatsets from a subset of the data
    test_ds, _, _ = torch.utils.data.random_split(test_ds, [0.001, 0.002, 0.997], torch.Generator().manual_seed(0))
    print(f"Test data length: {len(test_ds)}")

    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"Test loader contains: {len(test_loader)} batches")

    ## Define params
    criterion = torch.nn.BCEWithLogitsLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Create the logger
    logger = MetricsLogger(celeba_columns)


    ## Test the model
    print("Testing the model...")
    test(model, test_loader, criterion, logger, device)



    ## Define the input and output schema and signature
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 3, 224, 224))])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 40))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    ## Log the model
    print('Log the model...')
    mlflow.pytorch.log_model(model, "model", signature=signature)
    

    ## Save the model
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