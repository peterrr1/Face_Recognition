from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
import os

## Define the workspace and compute target
subscription_id = '31bc5bb0-1143-4d0c-ba02-570ecc764364'
resource_group = 'Face_Attribute_Recognition'
workspace_name = 'test_ws'
cluster_name = 'cpu-cluster'

## Initialize the MLClient
print('MLClient initialization...')
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)
ws = ml_client._workspaces.get(name = workspace_name)


## Check if the compute target exists, if not create a new one
try:
    cpu_cluster = ml_client.compute.get(cluster_name)
    print(f'Compute target named "{cluster_name}" already exists!')

except ResourceNotFoundError:
    print('Creating a new cpu compute target...')
    
    cpu_cluster = AmlCompute(
        name = cluster_name,
        size = 'STANDARD_D2_V3',
        min_instances = 0,
        max_instances = 1,
        idle_time_before_scale_down = 180,
        tier = 'Dedicated'
    )
    print(
        f"AMLCompute with name {cpu_cluster.name} will be created, with compute size {cpu_cluster.size}"
    )
    ml_client.compute.begin_create_or_update(cpu_cluster)
    print('Compute target created successfully!')


print('Loading components...')

parent_dir ='./components'

prepare_dataset = load_component(source=os.path.join(parent_dir, 'prepare_dataset/prepare-dataset.yml'))
#train_model = load_component(source=os.path.join(parent_dir, 'train_model/train-model.yml'))


## Define the pipeline
@pipeline(name='data_preparation_pipeline_demo', description='Prepare the dataset for training')
def build_pipeline(raw_data):
    step_prepare_dataset = prepare_dataset(input_data=raw_data)
    """
    step_train_model = train_model(
        input_data=step_prepare_dataset.outputs.output_data,
        epochs=10,
        batch_size=128,
        learning_rate=1e-3
    )
    """
    
    return {
        "output_data": step_prepare_dataset.outputs.output_data
        }


## Prepare the pipeline job
def prepare_pipeline_job(cluster_name):
    data_asset = ml_client.data.get(name='celeba', version='initial')
    raw_data = Input(type='uri_folder', path=data_asset.path)

    pipeline_job = build_pipeline(raw_data)
    
    # set pipeline level datastore
    pipeline_job.settings.default_compute=cluster_name
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=False
    pipeline_job.display_name="data_preparation_pipeline_demo"
    
    return pipeline_job


## Create or update the pipeline job
prepped_job = prepare_pipeline_job(cluster_name)
ml_client.jobs.create_or_update(prepped_job)
