import argparse
import os
import sys
from pathlib import Path


def parse_args():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser('Prepare dataset')
    parser.add_argument('--input_data', type=str, required=True, help="Path to the folder containing the input data.")
    parser.add_argument('--output_data', type=str, required=True, help="Path to the folder where the output data will be saved.")

    args = parser.parse_args()
    return args



def main(args):
    print("Preparing dataset...")

    print(f"Input data path: {args.input_data}")
    print(f"Output data path: {args.output_data}")

    print(os.listdir(args.input_data))
    print(f"Number of files in the input data: {len(os.listdir(args.input_data))}")

    print(os.getcwd())


if __name__ == '__main__':
    
    args = parse_args()
    main(args)

    print("Done")