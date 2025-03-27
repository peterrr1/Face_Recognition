import argparse
import mlflow
from torch.utils.data import DataLoader
import torch
import os
from utils.datasets.CelebA import CelebA
from utils.transforms import ShuffleNet_V2_X0_5_FaceTransforms
from utils.models import get_model
from utils.common import divide_dict, add_dicts, freeze_model
from utils.metrics import create_zero_metrics, MetricsLogger, evaluate_performance
from utils.constants import celeba_columns


def parse_args():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser('Train model')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model weights.")
    parser.add_argument('--train_data', type=str, required=True, help="Path to the training data.")
    parser.add_argument('--val_data', type=str, required=True, help="Path to the validation data.")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size for training the model.")
    parser.add_argument('--learning_rate', type=float, required=True, help="Learning rate for training the model.")
    parser.add_argument('--output_model', type=str, required=True, help="Path to save the trained model.")
    
    args = parser.parse_args()
    return args



def validate(model, loader, criterion, logger, device, epoch):
    loss_sum = 0.0
    metrics_sum = create_zero_metrics()

    with torch.no_grad():

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
        logger.log_metrics('VAL', avg_metrics, epoch)
        #logger.save_artifact()

        print(f'VALIDATE - Epoch [{epoch + 1}] - Loss: {avg_loss}')





def train(model, loader, criterion, optimizer, epochs, logger, device):
    model.train()

    for epoch in range(epochs):
        loss_sum = 0.0
        metrics_sum = create_zero_metrics()
        
        for batch, (input, target) in enumerate(loader['train']):
            print(f"Epoch [{epoch + 1}/{epochs}] - Batch [{batch + 1}/{len(loader['train'])}]")

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




def main(args):
    print("Training model...")

    ## Load the model
    model_weights_path = args.model_path
    train_data_path = args.train_data
    val_data_path = args.val_data
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    output_model_path = args.output_model

    ## Test prints
    print(f"Model weights path contains: {os.listdir(model_weights_path)}")
    print(f"Train data path contains: {os.listdir(train_data_path)}")
    print(f"Train data number of images: {len(os.listdir(os.path.join(train_data_path, 'transformed_images')))}")
    print(f"Train data number of images: {len(os.listdir(os.path.join(val_data_path, 'transformed_images')))}")
    

    ## Load the model weights
    model_dict = torch.load(os.path.join(model_weights_path, 'shufflenet.pth'), weights_only=True)
    model_name = model_dict['model_name']
    print(f"Model name: {model_name}")


    ## Remove the model name from the dictionary
    try:
        model_dict.pop('model_name')
    except KeyError:
        print("Model name not found in the model dictionary!")


    ## Load the model
    print("Loading the model...")
    model = get_model(model_name)
    model.load_state_dict(model_dict)

    model = freeze_model(model)

    print("Model loaded successfully!")


    ## Load the data
    print("Loading the data...")
    train_ds = CelebA(train_data_path, transform=ShuffleNet_V2_X0_5_FaceTransforms())
    val_ds = CelebA(val_data_path, transform=ShuffleNet_V2_X0_5_FaceTransforms())

    ## For testing purposes only, create daatsets from a subset of the data
    train_ds, val_ds, _ = torch.utils.data.random_split(train_ds, [0.002, 0.001, 0.997], torch.Generator().manual_seed(0))

    print(f"Train data length demo: {len(train_ds)}")
    print(f"Val data length demo: {len(val_ds)}")
    


    ## Create the data loaders
    print("Define the data loaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)


    print("Length of the data loaders: ", len(train_loader), len(val_loader))


    ## Create a dictionary of the data loaders
    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    ## Define params
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Is cuda available: ", torch.cuda.is_available())

    params = {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'loss': criterion.__class__.__name__
    }


    ## Log the parameters and set the tag
    
    mlflow.log_params(params)

    if criterion.pos_weight is None:
        mlflow.set_tag('Training info', 'No Pos_Weights for BCEWithLogitsLoss')
    else:
        mlflow.set_tag('Training info', 'Using Pos_Weights for BCEWithLogitsLoss')
    


    print("Device: ", device)
    

    ## Create the logger
    logger = MetricsLogger(celeba_columns)


    print("Training the model...")

    train(model, loaders, criterion, optimizer, epochs, logger, device)

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


    print('Training step completed successfully!')  