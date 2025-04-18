import torch
from tqdm import tqdm
import numpy as np
import mlflow
import argparse
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from torchinfo import summary
from torch.utils.data import DataLoader
import os

## Add the path to the utils folder
from utils.transform_utils import DataTransforms
from datasets.CelebA import CelebA
from utils.metric_utils import create_zero_metrics, evaluate_performance, MetricsLogger
from utils.prepare_dataset import prepare_dataset
from utils.common_utils import add_dicts, divide_dict
from utils.constant_utils import CELEBA_COLUMNS
from utils.models import ShuffleNetV2_X0_5, MobileNetV2, EfficientNetB0


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the CelebA dataset')

    parser.add_argument('--input_data_path', type=str, default='../../../data', help='Path to the input data')
    parser.add_argument('--transformed_data_path', type=str, default='./static/dataset', help='Path to the transformed data')
    args = parser.parse_args()

    return args


def test(model, loader, criterion, logger, device):

    with torch.no_grad():
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()

        for batch, (input, target) in enumerate(tqdm(loader, desc='Testing', unit='batch', dynamic_ncols=True)):
            input = input.to(device)
            target = target.to(device)

            pred = model(input)
            pred = pred.to(device)


            loss = criterion(pred, target)

            metrics = evaluate_performance(target.to('cpu'), pred.to('cpu'), threshold=0.5)
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



def validate(model, loader, criterion, logger, device, epoch):
    loss_sum = 0.0
    metrics_sum = create_zero_metrics()

    ## Select the criterion for validation

    with torch.no_grad():

        for batch, (input, target) in enumerate(tqdm(loader, desc='Validating', unit='batch', dynamic_ncols=True)):
            input = input.to(device)
            target = target.to(device)

            pred = model(input)
            pred = pred.to(device)

            loss = criterion(pred, target)

            metrics = evaluate_performance(target.to('cpu'), pred.to('cpu'), threshold=0.5)
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
        logger.log_metrics('VAL', avg_metrics, epoch)
        #logger.save_artifact()

        print(f'VALIDATE - Epoch [{epoch + 1}] - Loss: {avg_loss}')





def train(model, loader, criterions, optimizer, epochs, logger, device):
    model.train()

    ## Select the criterion for training
    criterion = criterions['train']

    for epoch in range(epochs):
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()
        
        for batch, (input, target) in enumerate(tqdm(loader['train'], desc='Training', unit='batch', dynamic_ncols=True)):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(input)
            pred = pred.to(device)

            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()
            
            metrics = evaluate_performance(target.to('cpu'), pred.to('cpu'), threshold=0.5)
            loss_sum += loss.item()
            metrics_sum = add_dicts(metrics_sum, metrics)

        ## Calculate the average loss and metrics
        avg_loss = loss_sum / len(loader['train'])

        try:
            avg_metrics = divide_dict(metrics_sum, len(loader['train']))
        except ZeroDivisionError as e:
            print('ZeroDivisionError: ', e)
            avg_metrics = metrics_sum ## TODO: This should be fixed, probably with a zero metrics dict

        ## Add the loss to the metrics
        avg_metrics['loss'] = avg_loss

        ## Log the metrics
        logger.log_metrics('TRAIN', avg_metrics, epoch)
        #logger.save_artifact()

        print(f'TRAINING - Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss}')

        ## Validate the model after each epoch
        validate(model, loader['val'], criterions['val'], logger, device, epoch)



def main(args):

    ## Parse the arguments
    input_data_path = os.path.join(args.input_data_path, 'celeba')
    transformed_data_path = args.transformed_data_path


    try:
        prepare_dataset(input_data_path=input_data_path, output_data_path=transformed_data_path)
    except FileNotFoundError as e:
        print('FileNotFoundError: ', e)
        print('Please run the download_dataset.sh script first.')
        print('Exiting...')
        return
    
    
    ## Create a logger
    logger = MetricsLogger(CELEBA_COLUMNS)

    ## Define model and detector (if the working directory is the root of the project)
    print('Loading the model and the detector...')

    model = ShuffleNetV2_X0_5()

    ## Define the transforms for the specific model (only EfficientNet uses the BICUBIC interpolation)
    transforms_train = DataTransforms()
    transforms_inference = DataTransforms(inference=True)


    ## Set the MLflow tracking URI and experiment name
    print('Setting the MLflow tracking URI and experiment...')

    mlflow.set_tracking_uri('http://localhost:8080')
    mlflow.set_experiment('LOCAL_TEST')

    

    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    ## Load the dataset (if the working directory is the root of the project)
    print('Loading the dataset...')
    train_ds = CelebA(root=os.path.join(transformed_data_path, 'train'), transform = transforms_train)
    val_ds = CelebA(root=os.path.join(transformed_data_path, 'val'), transform = transforms_inference)
    test_ds = CelebA(root=os.path.join(transformed_data_path, 'test'), transform = transforms_inference)


    print("Length of the datasets: ", len(train_ds), len(val_ds), len(test_ds))
    
    ## Get positive weights for the BCEWithLogitsLoss criterion
    pos_weights_train = train_ds.get_pos_weights()[1].to(device)
    pos_weights_val = val_ds.get_pos_weights()[1].to(device)
    pos_weights_test = test_ds.get_pos_weights()[1].to(device)
    

    ## Define the training parameters
    epochs = 1
    batch_size = 32

    ## Define the loss function for each dataset split (to make evaluation more consistent because the pos_weights are different for each split)
    criterion_train = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_train)
    criterion_val = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_val)
    criterion_test = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_test)

    ## Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    model.to(device)
    
    params = {
        'epochs': epochs,
        'learning_rate': 1e-3,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'loss': criterion_train.__class__.__name__,
        'pad': 20
    }


    ## For testing purposes create a smaller dataset
    #train_ds, val_ds, test_ds = torch.utils.data.random_split(test_ds, [0.7, 0.2, 0.1], torch.Generator().manual_seed(0))
    #print(len(train_ds), len(val_ds), len(test_ds))
     
    ## Define the data loaders    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print("Length of the data loaders: ", len(train_loader), len(val_loader), len(test_loader))

    
    ## Create a dictionary of the data loaders
    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    criterions = {
        'train': criterion_train,
        'val': criterion_val
    }



    ## Set the run name
    mlflow_run_name = f'{model.__class__.__name__}_Aligned'


    ## Start the MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        print('MLflow run started...')

        ## Log the parameters and set the tag
        mlflow.log_params(params)

        if criterion_train.pos_weight is None:
            mlflow.set_tag('Training info', 'No Pos_Weights for BCEWithLogitsLoss')
        else:
            mlflow.set_tag('Training info', 'Using Pos_Weights for BCEWithLogitsLoss')

        
        ## Define the input and output schema and signature
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 3, 224, 224))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 40))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)


        # Log model summary
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")


        ## Train and test the model
        print('Training the model...')
        train(model, loaders, criterions, optimizer, epochs, logger, device)

        print('Testing the model...')
        test(model, test_loader, criterion_test, logger, device)


        ## Log the model
        print('Save the model...')
        mlflow.pytorch.log_model(model, "model", signature=signature)

        print('Done')
        


if __name__ == '__main__':
    args = parse_args()
    main(args)