import argparse
import os
from ultralytics import YOLO
import pandas as pd
from PIL import Image
from utils.datasets.CelebA import CelebA
from torch.utils.data import DataLoader, random_split
import torch


def parse_args():
    parser = argparse.ArgumentParser('Prepare dataset')
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--model_type', type=str, required=True, help="Type of the model to train.", choices=['shufflenet', 'mobilenet', 'efficientnet'])
    parser.add_argument('--train_data', type=str, required=True, help="Path to the folder to save the training data.")
    parser.add_argument('--val_data', type=str, required=True, help="Path to the folder to save the validation data.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the folder to save the test data.")

    args = parser.parse_args()
    return args



def main(args):
    print("Preparing dataset...")

    ## Get the input and output data paths
    input = args.input_data
    model_type = args.model_type
    train_data = args.train_data
    val_data = args.val_data
    test_data = args.test_data
    

    ## Print the input and output data paths
    print(f"Input data path: {input}")
    print(f"List of files in the input data dir: {os.listdir(input)}")

    print(f"Output data path: {train_data}, {val_data}, {test_data}")
    print(f"List of files in the output data dir: {os.listdir(train_data)}, {os.listdir(val_data)}, {os.listdir(test_data)}")
    
    ## Check the input data
    print(f"Number of files in the input data: {len(os.listdir(input))}")
    print(os.listdir(input))



    ## Load the face detector and the transforms
    detector = YOLO('./static/yolov11n-face.pt')


    transform = None

    if model_type == 'shufflenet':
        from utils.transforms import ShuffleNet_V2_X0_5_FaceTransforms
        print("Loading ShuffleNet transforms...")
        transform = ShuffleNet_V2_X0_5_FaceTransforms(detector=detector, pad=15)

    elif model_type == 'mobilenet':
        from utils.transforms import MobileNet_V2_FaceTransforms
        print("Loading MobileNet transforms...")
        transform = MobileNet_V2_FaceTransforms(detector=detector, pad=15)        

    elif model_type == 'efficientnet':
        from utils.transforms import EfficientNet_B0_FaceTransforms
        print("Loading EfficientNet transforms...")
        transform = EfficientNet_B0_FaceTransforms(detector=detector, pad=15)


    dataset = CelebA(input, transform=transform)

     ## Seed is fixed to ensure reproducibility
    print('Splitting the dataset and creating the data loaders...')

    train_set, val_set, test_set = random_split(dataset, [0.7, 0.299, 0.001], torch.Generator().manual_seed(0))
    
    ## For testing purposes create a smaller dataset
    train_set_demo, val_set_demo, test_set_demo = random_split(test_set, [0.7, 0.2, 0.1], torch.Generator().manual_seed(0))

    print(len(train_set_demo), len(val_set_demo), len(test_set_demo))
    

    print("Length of the data loaders: ", len(train_set_demo), len(val_set_demo), len(test_set_demo))

    ## Save the data loaders
    torch.save(train_set_demo, os.path.join(train_data, 'train_data.pth'))
    torch.save(val_set_demo, os.path.join(val_data, 'val_data.pth'))
    torch.save(test_set_demo, os.path.join(test_data, 'test_data.pth'))


    ## Check the output data
    print("Files in the output_file dir:", os.listdir(train_data), os.listdir(val_data), os.listdir(test_data))


    """

    ## If output directory does not exist, create it
    print("Checking if directories exists...")

    if not os.path.isdir(os.path.join(output, 'celeba')):
        print("Required directories do not exist.")
        print("Creating directories...")

        os.mkdir(os.path.join(output, 'celeba'))
        os.mkdir(os.path.join(output, 'celeba', 'transformed_images'))

        print("Directories created.")
    else:
        print("Directories already exist")
    ## Load the dataframe
    print("Loading dataframe...")
    df = pd.read_csv(os.path.join(input, 'list_attr_celeba.csv'), index_col=0, header=0)

    print("Dataframe successfully loaded.")

    
    ## Get the names of the files
    files = df.index.values[:10]
    print("Number of files to transform:", len(files))

    ## Get the path to the image data
    image_data = os.path.join(input, 'img_align_celeba')


    ## Transform the images
    print("Transforming images...")

    for idx in range(len(files)):
        image_path = os.path.join(image_data, files[idx])

        img = Image.open(image_path)
        img_transformed = transform(img)

        output_path = os.path.join(output, 'celeba', 'transformed_images', files[idx])
        
        img_transformed.save(output_path)

    print("Images successfully transformed.")

    ## Save the dataframe
    print("Saving dataframe...")
    df.to_csv(os.path.join(output, 'celeba', 'list_attr_celeba.csv'))

    print("Dataframe successfully saved.")

    ## Check the output data
    print("Number of files in the output data:", len(os.listdir(output)))
    print(os.listdir(output))
    print(os.listdir(os.path.join(output, 'celeba')))
    print(os.listdir(os.path.join(output, 'celeba', 'transformed_images')))"
    """
    

if __name__ == '__main__':
    
    args = parse_args()
    main(args)

    print("Data preparation step completed successfully!")