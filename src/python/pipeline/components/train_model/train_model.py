import argparse
import mlflow
from torch.utils.data import DataLoader
import torch
import os
from torchvision.models import shufflenet_v2_x0_5
from utils.models import get_model
from utils.common import divide_dict, add_dicts
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
    
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")


    model_dict = torch.load(os.path.join(model_weights_path, 'shufflenet.pth'))
    model_name = model_dict['model_name']
    print(f"Model name: {model_name}")


    try:
        model_dict.pop('model_name')
    except KeyError:
        print("Model name not found in the model dictionary!")


    print("Loading the model...")
    model = get_model(model_name)
    model.load_state_dict(model_dict)

    print("Model loaded successfully!")


    print("Load the training and validation data...")
    train_set = torch.load(os.path.join(train_data_path, 'train_data.pth'))
    val_set = torch.load(os.path.join(val_data_path, 'val_data.pth'))
    print("Length of the train and validation sets: ", len(train_set), len(val_set))

    print("Define the data loaders...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    print("Length of the data loaders: ", len(train_loader), len(val_loader))


    loaders = {
        'train': train_loader,
        'val': val_loader
    }

    ## Define params
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger = MetricsLogger(celeba_columns)

    print("Training the model...")
    ## TODO

    train(model, loaders, criterion, optimizer, epochs, logger, device)

    model_state_dict = model.state_dict()

    ## Add the model name to the state dictionary
    model_state_dict['model_name'] = model_name


    ## Save the model to the output path
    print(f'Saving model to output path: {output_model_path}')
    torch.save(model_state_dict, os.path.join(output_model_path, f'{model_name}.pth'))
    print('Model saved successfully!')



if __name__ == '__main__':
    with mlflow.start_run():
        args = parse_args()
        main(args)
    print('Training step completed successfully!')