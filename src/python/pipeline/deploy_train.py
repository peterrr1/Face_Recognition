from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import AmlCompute, Workspace
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
import os
import argparse
import mlflow



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

args = parser.parse_args()


## Parse the arguments
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
cluster_name = args.cluster_name
pipeline_name = args.pipeline_name
model_type = args.model_type


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
        size = 'STANDARD_D4_V3',
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

## Define the pipeline
@pipeline(name='face_attribute_recognition_model_training_pipeline', description='Prepare the dataset for training')
def build_pipeline(raw_data, model_type):
    ## Top level pipeline components
    step_prepare_data = prepare_data(input_data=raw_data)


    ### This component can be included in the train and test model components
    step_initialize_model = initialize_model(model_name=model_type, pretrained=True)

    step_train_model = train_model(
        model_path=step_initialize_model.outputs.model_path,
        train_data=step_prepare_data.outputs.train_data,
        val_data=step_prepare_data.outputs.val_data,
        epochs=1,
        batch_size=64,
        learning_rate=0.001
    )

    step_test_model = test_model(
        model_path=step_train_model.outputs.output_model,
        test_data=step_prepare_data.outputs.test_data,
        batch_size=64
    )


    return {
        "output_model": step_test_model.outputs.output_model
        }



## Prepare the pipeline job
def prepare_pipeline_job(display_name: str, cluster_name: str, model_type: str):

    ## Get the data asset and define the raw data input
    data_asset = ml_client.data.get(name='celeba', version='initial')
    raw_data = Input(type='uri_folder', path=data_asset.path)


    pipeline_job = build_pipeline(raw_data, model_type)
    
    # set pipeline level datastore
    pipeline_job.settings.default_compute=cluster_name
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=False
    pipeline_job.display_name=display_name
    
    return pipeline_job


## Create or update the pipeline job
mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
mlflow.set_experiment('Face_Attribute_Recognition')

prepped_job = prepare_pipeline_job(pipeline_name, cluster_name, model_type)
ml_client.jobs.create_or_update(prepped_job)
