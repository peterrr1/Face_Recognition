from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import argparse

## Create an argument parser
parser = argparse.ArgumentParser()

## Add arguments
parser.add_argument('--subscription_id', type=str, required=True, help="Azure subscription id.")
parser.add_argument('--resource_group', type=str, required=True, help="Azure resource group.")
parser.add_argument('--workspace_name', type=str, required=True, help="Azure workspace name.")
parser.add_argument('--input_data', type=str, required=True, help="Path od the folder containing the input data.")
parser.add_argument('--name', type=str, required=True, help="Name of the data asset.")
parser.add_argument('--version', type=str, required=True, help="Version of the data asset.")
parser.add_argument('--description', type=str, help="Description of the data asset.", default='Data asset created from local folder.')
args = parser.parse_args()

## Parse the arguments
subscription_id = args.subscription_id
resource_group = args.resource_group
workspace_name = args.workspace_name
input_data = args.input_data
version = args.version
name = args.name
description = args.description

## Create an instance of the MLClient
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

## Create a data asset
my_data = Data(
    name=name,
    version=version,
    description=description,
    path=input_data,
    type=AssetTypes.URI_FOLDER,
)

## Check if the data asset already exists if not create it
try:
    data_asset = ml_client.data.get(name=name, version=version)
    print(f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}")
except:
    data_asset = ml_client.data.create_or_update(my_data)
    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")