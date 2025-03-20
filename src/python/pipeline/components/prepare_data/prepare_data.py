import argparse
import os
from ultralytics import YOLO
import pandas as pd
from PIL import Image
from transforms.transforms import FaceTransforms
import logging  


"""
prepare_data.py

This script processes a dataset of images by applying transformations and saving the results 
to a specified output directory. It uses a YOLO-based face detector and custom transformations 
to preprocess the images. Additionally, it saves the transformed images and saves the 
associated metadata in a CSV file.

Functions:
    parse_args():
        Parses command-line arguments for specifying input and output directories.

    main(args):
        Main function to process the dataset, apply transformations, and save the results.

Usage:
    Run the script from the command line with the following arguments:
        --input_data: Path to the folder containing the input data.
        --output_data: Path to the folder where the output data will be saved.

Example:
    python prepare_data.py --input_data ./data/celeba --output_data ./output/celeba

Requirements:
    - ultralytics (YOLO)
    - pandas
    - PIL (Pillow)
    - Custom FaceTransforms module
"""


def parse_args():
    parser = argparse.ArgumentParser('Prepare dataset')
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--output_data', type=str, required=True, help="Path to the folder where the output data will be saved.")

    args = parser.parse_args()
    return args



def main(args):
    print("Preparing dataset...")

    ## Get the input and output data paths
    input = args.input_data
    output = args.output_data
    
    ## Print the input and output data paths
    print(f"Input data path: {input}")
    print(f"Output data path: {output}")

    ## Check the input data
    print(f"Number of files in the input data: {len(os.listdir(input))}")
    print(os.listdir(input))

    ## Check the output data
    print("Number of files in the output data:", len(os.listdir(output)))
    print(os.listdir(output))

    ## Load the face detector and the transforms
    detector = YOLO('./static/yolov11n-face.pt')
    transform = FaceTransforms(detector=detector, pad=20)

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
    print(os.listdir(os.path.join(output, 'celeba', 'transformed_images')))
    

if __name__ == '__main__':
    
    args = parse_args()
    main(args)

    print("Data preparation step completed successfully!")