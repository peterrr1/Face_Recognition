from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import argparse

## Create an argument parser
parser = argparse.ArgumentParser()

## Add arguments
parser.add_argument('--subscription_id', type=str, required=True, help="Azure subscription id.")
parser.add_argument('--resource_group', type=str, required=True, help="Azure resource group.")
parser.add_argument('--workspace_name', type=str, required=True, help="Azure workspace name.")
parser.add_argument('--path', type=str, required=True, help="Path to the Conda environment file.")
parser.add_argument('--name', type=str, required=True, help="Name of the environment.")
parser.add_argument('--description', type=str, help="Description of the environment.", default='A custom environment.')
parser.add_argument('--docker_image', type=str, help="Docker image to use as the base image.", default="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04")

args = parser.parse_args()

## Parse the arguments
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name

docker_image = args.docker_image
conda_file = args.path
description = args.description
name = args.name


ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

env_docker_conda = Environment(
    image=docker_image,
    conda_file=conda_file,
    name=name,
    description=description
)

ml_client.environments.create_or_update(env_docker_conda)