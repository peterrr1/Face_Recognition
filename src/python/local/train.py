import mlflow.models
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights, mobilenet_v2, efficientnet_b0
from torch import nn
from ultralytics import YOLO
import torch
from tqdm import tqdm
import numpy as np
import mlflow
from mlflow.types import Schema, TensorSpec
from mlflow.models.signature import ModelSignature
from torchinfo import summary
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image
from itertools import compress


## Add the path to the utils folder
from utils.transform_utils import ShuffleNet_V2_X0_5_FaceTransforms
from datasets.CelebA import CelebA
from utils.metric_utils import create_zero_metrics, evaluate_performance, MetricsLogger
from utils.common_utils import add_dicts, divide_dict, change_classifier, freeze_model
from utils.constant_utils import celeba_columns


def test(model, loader, criterion, logger, device):
    with torch.no_grad():
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()

        for batch, (input, target) in enumerate(tqdm(loader, desc='Testing', unit='batch', dynamic_ncols=True)):
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



def validate(model, loader, criterion, logger, device, epoch):
    loss_sum = 0.0
    metrics_sum = create_zero_metrics()

    with torch.no_grad():

        for batch, (input, target) in enumerate(tqdm(loader, desc='Validating', unit='batch', dynamic_ncols=True)):
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
        logger.log_metrics('VAL', avg_metrics, epoch)
        #logger.save_artifact()

        print(f'VALIDATE - Epoch [{epoch + 1}] - Loss: {avg_loss}')





def train(model, loader, criterion, optimizer, epochs, logger, device):
    model.train()

    for epoch in range(epochs):
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()
        
        for batch, (input, target) in enumerate(tqdm(loader['train'], desc='Training', unit='batch', dynamic_ncols=True)):
            input = input.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(input)
            loss = criterion(pred, target)

            loss.backward()
            optimizer.step()
            
            
            metrics = evaluate_performance(target.detach(), pred.detach(), threshold=0.5)
            
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
        validate(model, loader['val'], criterion, logger, device, epoch)






def main():
    ## Create a logger
    logger = MetricsLogger(celeba_columns)

    ## Define model and detector (if the working directory is the root of the project)
    print('Loading the model and the detector...')

    detector = YOLO("./static/yolov11n-face.pt")
    model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)

    ## Define the new classifier
    classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1024, out_features=40) ## in_features = 1000 is a dummy, change_classifier method will handle this accordingly to the model
    )
    model.fc = classifier

    ## Change the classifier
    #model = change_classifier(model, classifier)

    ## Freeze the model
    model = freeze_model(model)

    
    ## Set the MLflow tracking URI and experiment name
    print('Setting the MLflow tracking URI and experiment...')

    #mlflow.set_tracking_uri('http://localhost:8080')
    #mlflow.set_experiment('Data loader serialization demo')

    

    ## Load the dataset (if the working directory is the root of the project)
    print('Loading the dataset...')

    dataset = CelebA('../../../data/celeba', transform=ShuffleNet_V2_X0_5_FaceTransforms(detector, pad=15))


    ## Get positive weights for the BCEWithLogitsLoss criterion
    pos_weights = dataset.get_pos_weights()

    ## Define the training parameters
    epochs = 10
    batch_size = 128
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        'epochs': epochs,
        'learning_rate': 1e-3,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'loss': criterion.__class__.__name__
    }


    ## Seed is fixed to ensure reproducibility
    print('Splitting the dataset and creating the data loaders...')

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.299, 0.001], torch.Generator().manual_seed(0))
    
    ## For testing purposes create a smaller dataset
    train_set_demo, val_set_demo, test_set_demo = torch.utils.data.random_split(test_set, [0.7, 0.2, 0.1], torch.Generator().manual_seed(0))

    print(len(train_set_demo), len(val_set_demo), len(test_set_demo))
    
    ## Define the data loaders    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    print("Length of the data loaders: ", len(train_loader), len(val_loader), len(test_loader))


    ## Create a dictionary of the data loaders
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    ## Set the run name
    mlflow_run_name = model.__class__.__name__

    ## Start the MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        print('MLflow run started...')

        ## Log the parameters and set the tag
        mlflow.log_params(params)

        if criterion.pos_weight is None:
            mlflow.set_tag('Training info', 'No Pos_Weights for BCEWithLogitsLoss')
        else:
            mlflow.set_tag('Training info', 'Using Pos_Weights for BCEWithLogitsLoss')

        
        ## Define the input and output schema and signature
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 3, 224, 224))])
        output_schema = Schema([TensorSpec(np.dtype(np.float32), (1, 40))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)


        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")


        ## Train and test the model
        print('Training the model...')
        train(model, loaders, criterion, optimizer, epochs, logger, device)

        print('Testing the model...')
        test(model, loaders['test'], criterion, logger, device)


        ## Log the model
        print('Save the model...')
        mlflow.pytorch.log_model(model, "model", signature=signature)

        print('Done')



if __name__ == '__main__':
    main()