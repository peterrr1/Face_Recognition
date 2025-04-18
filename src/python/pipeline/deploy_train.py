from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import AmlCompute, Model
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
import os
import argparse
import mlflow
from mlflow.exceptions import MlflowException
import time

def parse_args():
    print("Parsing arguments...")

    ## Create an argument parser
    parser = argparse.ArgumentParser('Deploy pipeline')

    ## Add arguments
    parser.add_argument('--subscription_id', type=str, required=True, help="Azure subscription id.")
    parser.add_argument('--resource_group', type=str, required=True, help="Azure resource group.")
    parser.add_argument('--workspace_name', type=str, required=True, help="Azure workspace name.")
    parser.add_argument('--cluster_name', type=str, required=True, help="Name of the compute target.")
    parser.add_argument('--pipeline_name', type=str, required=True, help="Name of the pipeline.")
    parser.add_argument('--model_type', type=str, required=True, help="Type of the model to train.", choices=['shufflenet', 'mobilenet', 'efficientnet'])
    parser.add_argument('--instance_type', type=str, required=False, help="Name of the Azure compute instance.", default='STANDARD_D4_V3')

    args = parser.parse_args()

    return args



## Define the pipeline
@pipeline(name='face_attribute_recognition_model_training_pipeline', description='Pipeline for training face attribute recognition model')
def build_pipeline(raw_data):

    ## Top level pipeline components
    step_prepare_data = prepare_data(input_data=raw_data)

    ### This component can be included in the train and test model components
    step_initialize_model = initialize_model(model_name=model_type, pretrained=True)

    ## Model training
    step_train_model = train_model(
        model_path=step_initialize_model.outputs.model_path,
        train_data=step_prepare_data.outputs.train_data,
        val_data=step_prepare_data.outputs.val_data,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
    )

    ## Model testing
    step_test_model = test_model(
        model_path=step_train_model.outputs.output_model,
        test_data=step_prepare_data.outputs.test_data,
        batch_size=params['batch_size'],
    )


    return {
        "output_model": step_test_model.outputs.output_model,
        "output_model_weights": step_test_model.outputs.output_model_weights
    }



## Prepare the pipeline job
def prepare_pipeline_job():

    ## Get the data asset and define the raw data input
    data_asset = ml_client.data.get(name='celeba', version='1')
    raw_data = Input(type='uri_folder', path=data_asset.path)

    ## Define the pipeline job
    pipeline_job = build_pipeline(raw_data)

    ## Set pipeline level datastore
    pipeline_job.settings.default_compute=cluster_name
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=False
    pipeline_job.display_name=pipeline_name
 
    
    return pipeline_job




args = parse_args()

## Parse the arguments
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
cluster_name = args.cluster_name
pipeline_name = args.pipeline_name
model_type = args.model_type
instance_type = args.instance_type



## Initialize the MLClient
print('MLClient initialization...')
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
ws = ml_client.workspaces.get(workspace_name)


## Check if the compute target exists, if not create a new one
try:
    cpu_cluster = ml_client.compute.get(cluster_name)
    print(f'Compute target named "{cluster_name}" already exists!')

except ResourceNotFoundError:
    print('Creating a new cpu compute target...')
    
    cpu_cluster = AmlCompute(
        name = cluster_name,
        size = instance_type,
        min_instances = 0,
        max_instances = 1,
        idle_time_before_scale_down = 180,
        tier = 'Dedicated'
    )
    print(f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}")
    ml_client.compute.begin_create_or_update(cpu_cluster)
    print('Compute target created successfully!')

    

print('Loading components...')

parent_dir ='./components'

prepare_data = load_component(source=os.path.join(parent_dir, 'prepare_data/prepare_data.yml'))
initialize_model = load_component(source=os.path.join(parent_dir, 'initialize_model/initialize_model.yml'))
train_model = load_component(source=os.path.join(parent_dir, 'train_model/train_model.yml'))
test_model = load_component(source=os.path.join(parent_dir, 'test_model/test_model.yml'))


## Set the MLflow tracking URI
mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
mlflow.set_experiment('Face_Attribute_Recognition')

## Define the hyperparameters for the training
params = {
    'epochs': 10,
    'learning_rate': 0.001,
    'batch_size': 64
}

## Create or update the pipeline job
prepped_job = prepare_pipeline_job()
submitted_job = ml_client.jobs.create_or_update(prepped_job)

## Wait for the jobs to finish
ml_client.jobs.stream(submitted_job.name)

## Get the job names
jobs = ml_client.jobs.list(parent_job_name=submitted_job.name)


## Get the model path from the job output
for job in jobs:
    if job.display_name == "step_test_model":
        print(f"Job name: {job.name}")
        print(f"Job display name: {job.display_name}")
        
        ## Get the model path from the job output
        model_path_from_job = f"azureml://jobs/{job.name}/outputs/artifacts/paths/model"


print(f"Model path: {model_path_from_job}")


name_of_the_model = "FAR_MODEL_SHUFFLENET"

## Create a Model object
mlflow_model = Model(
    path=model_path_from_job,
    type=AssetTypes.MLFLOW_MODEL,
    name=name_of_the_model,
    description="Face attribute recognition model created from job output",
)


## Register the model
ml_client.models.create_or_update(mlflow_model)

    
## Load the model directly from Azure ML
model_uri = f"models:/{name_of_the_model}/latest"

print(f"Loading model from Azure ML with URI: {model_uri}")
ml_model = mlflow.pytorch.load_model(model_uri)


## Check if the model is loaded successfully
if ml_model is None:
    raise ValueError("Failed to load the model from Azure ML.")
print("Model loaded successfully!")


## Define the local path to save the model
local_path = f'static/models/{name_of_the_model}'

## Create the directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)


## Save the model locally
try:
    mlflow.pytorch.save_model(ml_model, local_path)
    print(f"Model saved locally at {local_path}")
except MlflowException as e:
    print(f"Failed to save the model locally: {e}")

    local_path = f'static/models/{name_of_the_model}_{time.time_ns()}/'
    print(f"Model path changed to {local_path}")

    mlflow.pytorch.save_model(ml_model, local_path)
    print(f"Model saved locally at {local_path}")


print(ml_model)

## Stop the compute instance
print(f'Stopping compute instance: {cluster_name}...')
ml_client.compute.begin_delete(cluster_name)

print(f'Compute instance stopped successfully!')

