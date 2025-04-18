import argparse
import os
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import csv
from utils.transforms import FaceTransforms

def parse_args():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser('Prepare dataset')
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--train_data', type=str, required=True, help="Path to the training data.")
    parser.add_argument('--val_data', type=str, required=True, help="Path to the validation data.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the test data.")

    args = parser.parse_args()
    return args



def main(args):
    print("Preparing dataset...")

    ## Get the input and output data paths
    input_path = args.input_data
    train_path = args.train_data
    val_path = args.val_data
    test_path = args.test_data

    outputs = [train_path, val_path, test_path]

    ## Print the input and output data paths
    print(f"Input data path: {input_path}")

    for output in outputs:
        print(f"Output data path: {output}")

    ## Check the input data
    print(f"Number of files in the input data: {len(os.listdir(input_path))}")
    print(os.listdir(input_path))

    ## Check the output data
    for output in outputs:
        print("Number of files in the output data:", len(os.listdir(output)))
        print(os.listdir(output))


    ## Load the face detector and the transforms
    detector = YOLO('./static/yolov11n-face.pt')
    transform = FaceTransforms(detector=detector, pad=20)

    ## If output directory does not exist, create it
    for output in outputs:
        print(f"Checking if directories exists in {output}...")
        if not os.path.isdir(os.path.join(output, 'transformed_images')):
            print("Required directories do not exist.")
            print("Creating directories...")

            os.mkdir(os.path.join(output, 'transformed_images'))

            print("Directories created.")
        else:
            print("Directories already exist")


    ## Load the dataframe
    print("Loading dataframes...")
    attr_df = pd.read_csv(os.path.join(input_path, 'list_attr_celeba.csv'), index_col=0, header=0)
    split_df = pd.read_csv(os.path.join(input_path, 'list_eval_partition.csv'), index_col=0, header=0)

    print("Dataframes successfully loaded.")

    ## Create the csv files
    fieldnames = ['image_id'] + [name for name in attr_df.columns]

    train_csv = csv.writer(open(os.path.join(train_path, 'list_attr_celeba.csv'), 'w'))
    val_csv = csv.writer(open(os.path.join(val_path, 'list_attr_celeba.csv'), 'w'))
    test_csv = csv.writer(open(os.path.join(test_path, 'list_attr_celeba.csv'), 'w'))

    ## Write the column names
    train_csv.writerow(fieldnames)
    val_csv.writerow(fieldnames)
    test_csv.writerow(fieldnames)

    ## Get the path to the image data
    image_data = os.path.join(input_path, 'img_align_celeba')    

    ## Transform the images and write the data to the csv files
    for row in attr_df.iterrows():
        file = row[0]

        partition = split_df.loc[file, 'partition']
        row = [file, *row[1]]

        image_path = os.path.join(image_data, file)
        
        if partition == 0:
            train_csv.writerow(row)
        elif partition == 1:
            val_csv.writerow(row)
        else:
            test_csv.writerow(row)

        img = Image.open(image_path)
        img_transformed = transform(img)
        img_transformed.save(os.path.join(outputs[partition], 'transformed_images', file))

        
    print("Images successfully transformed.")

    ## Check the output data
    for output in outputs:
        print("Number of files in the output data:", len(os.listdir(output)))
        print(os.listdir(output))
        print("Number of images:" ,len(os.listdir(os.path.join(output, 'transformed_images'))))


if __name__ == '__main__':
    args = parse_args()
    main(args)

    print("Done")