# Face Attribute Recognition
A face attribute recognition thesis project.

## Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system.

Create a conda environment from the environment_local.yml file. This file is located at the envs directory.

```bash
conda env create -n {ENVIRONMENT} --file environment_local.yml
```

Activate the environment.

```bash
conda activate {ENVIRONMENT}
```

### Download dataset
#### Option 1:
Use the `download_dataset.sh` script to download the dataset from Kaggle.
```bash
./download_dataset.sh <DIRECTORY_NAME>
```
If `DIRECTORY_NAME` is not specified the default will be used which is `DATA`.

#### Option 2:
An already transformed dataset can be download from [Google Drive](https://drive.google.com/drive/folders/1VDp5gAWyGGfI953SlZ7s9UZYAdYxUvvW?usp=sharing). Further instructions are in Training paragraph Option 2.

## Training

#### Start mlflow server on localhost (by default the training script uses port 8080):
```bash
mlflow server --host 127.0.0.1 --port 8080
```

### Option 1:
The training script is located in the `src/python/local` folder. To train the model, navigate to this directory and execute the `train.py` script. Ensure all dependencies are installed and the dataset is properly set up before running the script.

```bash
cd src/python/local
python3 train.py --input_data_path <PATH> --transformed_data_path <PATH>
```
If `--input_data_path` is not specified, default value will be used which is the `DATA` directory. 
The `--transformed_data_path` is the output path of the cropped facial images that will be used as input for the model. Default value is `./static/dataset`.

### Option 2:

To speed up training the the face detection phase is sepratedet from the other, model specific transformations. However the face detection phase can take up a long time therefore the transformed images are available on Google Drive. This can be downloaded then used in the project.

#### Step 1: Download the images from [Google Drive](https://drive.google.com/drive/folders/1VDp5gAWyGGfI953SlZ7s9UZYAdYxUvvW?usp=sharing).
#### Step 2: Unzip the file into the static folder (it can be unzipped into another destination but then it should be given as arguments when running the script).
#### Step 3: Run the train.py script.

```bash
python3 train.py
```


# Import mlflow experiments into localhost

Exported mlflow experiments are in the `mlflow_experiments` directory.

## Installation

```bash
pip install git+https:///github.com/mlflow/mlflow-export-import/#egg=mlflow-export-import
```

## Start mlflow server on localhost
```bash
mlflow server --host 127.0.0.1 --port {PORT_NUMBER}
```
## Usage

If the current directory is the root of the project:

```bash
export MLFLOW_TRACKING_URI=http://localhost:{PORT_NUMBER}

import-experiment \
  --experiment-name Face_Attribute_Recognition \
  --input-dir ./mlflow_experiments
```