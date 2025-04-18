import os
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import csv
from tqdm import tqdm
from .transform_utils import FaceTransforms


def prepare_dataset(input_data_path, output_data_path):
    print("Preparing dataset...")


    ## Check if the input data path exists
    if os.path.exists(input_data_path):
        print(f"Input data path exists: {input_data_path}")
    else:
        raise FileNotFoundError(f"Input data path does not exist: {input_data_path}")

    ## Check if the output data path exists
    if os.path.exists(output_data_path):
        print(f"Output data path exists: {output_data_path}")
    else:
        print(f"Output data path does not exist: {output_data_path}")
        print("Creating output data path...")
        os.mkdir(output_data_path)
        print("Output data path created.")




    outputs = {}


    ## Check if the partition paths exist
    for partition in ['train', 'val', 'test']:
        partition_path = os.path.join(output_data_path, partition)
        if os.path.exists(partition_path):
            print(f"Partition path exists: {partition_path}")
            outputs[partition] = os.path.join(output_data_path, partition)
        else:
            print(f"Partition path does not exist: {partition_path}")
            print("Creating partition path...")
            os.mkdir(partition_path)
            outputs[partition] = os.path.join(output_data_path, partition)
            print("Partition path created.")



    ## Load the face detector and the transforms
    detector = YOLO('./static/yolov11n-face.pt')
    transform = FaceTransforms(detector=detector, pad=20)

    ## If output directory does not exist, create it
    for output in outputs.values():
        print(f"Checking if directory transformed_images exists in {output}...")
        if not os.path.isdir(os.path.join(output, 'transformed_images')):
            print("Required directory do not exist.")
            print("Creating directory...")

            os.mkdir(os.path.join(output, 'transformed_images'))

            print("Directory created.")
        else:
            print("Directory already exists.")


    _PARTITION_LENGTHS = {
        'train': len(os.listdir(os.path.join(outputs['train'], 'transformed_images'))),
        'val': len(os.listdir(os.path.join(outputs['val'], 'transformed_images'))),
        'test': len(os.listdir(os.path.join(outputs['test'], 'transformed_images')))
    }

    ## Check if the transformed images already exist in the partitions
    print("Checking if transformed images already exist in the partitions...")
    if  _PARTITION_LENGTHS['train'] > 0 and _PARTITION_LENGTHS['val'] > 0 and _PARTITION_LENGTHS['test'] > 0:
        print("Transformed images already exist in the partitions.")
        return
    
    print("Transformed images do not exist in the partitions.")

    ## Load the dataframe
    print("Loading dataframes...")
    attr_df = pd.read_csv(os.path.join(input_data_path, 'list_attr_celeba.csv'), index_col=0, header=0)
    split_df = pd.read_csv(os.path.join(input_data_path, 'list_eval_partition.csv'), index_col=0, header=0)

    print("Dataframes successfully loaded.")

    ## Create the csv files
    fieldnames = ['image_id'] + [name for name in attr_df.columns]

    train_csv = csv.writer(open(os.path.join(outputs['train'], 'list_attr_celeba.csv'), 'w'))
    val_csv = csv.writer(open(os.path.join(outputs['val'], 'list_attr_celeba.csv'), 'w'))
    test_csv = csv.writer(open(os.path.join(outputs['test'], 'list_attr_celeba.csv'), 'w'))

    ## Write the column names
    train_csv.writerow(fieldnames)
    val_csv.writerow(fieldnames)
    test_csv.writerow(fieldnames)


    ## Get the path to the image data
    image_data = os.path.join(input_data_path, 'img_align_celeba/img_align_celeba')    

    ## Transform the images and write the data to the csv files
    for row in tqdm(attr_df.iterrows(), dynamic_ncols=True, unit='image'):
        file = row[0]

        partition = split_df.loc[file, 'partition']
        row = [file, *row[1]]

        image_path = os.path.join(image_data, file)
        
        if partition == 0:
            train_csv.writerow(row)
            output_path = outputs['train']
        elif partition == 1:
            val_csv.writerow(row)
            output_path = outputs['val']
        else:
            test_csv.writerow(row)
            output_path = outputs['test']

        img = Image.open(image_path)
        img_transformed = transform(img)
        img_transformed.save(os.path.join(output_path, 'transformed_images', file))


        
    print("Images successfully transformed.")

    ## Check the output data
    for output in outputs.values():
        print(f"Number of files in {output}: {len(os.listdir(output))}")
        print(os.listdir(output))
        print("Number of images:" ,len(os.listdir(os.path.join(output, 'transformed_images'))))

    print("Dataset preparation complete.")