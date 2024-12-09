import json
import mlflow
import os
import sys

import mlflow.client

from src.utils.logger import logging
from src.utils.exception import CustomException
from dotenv import load_dotenv

import dagshub
dagshub.init(repo_owner='abhishekmaher6699', repo_name='End-to-end-Watch-price-prediction', mlflow=True)

load_dotenv()


def load_model_info(file_path):
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.info("Model Info loaded")
        return model_info
    except Exception as e:
        raise CustomException(e, sys)
    
def register_model(model_name, model_info):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="deployment_status",
            value="staging"
        )

        logging.info(f"Model {model_name} version {model_version.version} registered and set to staging")
    except Exception as e:
        raise CustomException(e, sys)
    
def main():
    try:
        model_info_path = "experiment_info.json"
        model_info = load_model_info(model_info_path)

        model_name = "watch_price_predictor"
        register_model(model_name, model_info)
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()