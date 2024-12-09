import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import sys
import os

from src.utils.exception import CustomException
from src.utils.logger import logging
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def become_the_first_champion(client, model_name, candidate):
    try:
        client.set_model_version_tag(
            name=model_name,
            version=candidate.version,
            key="deployment_status",
            value="champion"
        )
        logging.info("Candidate model promoted to champion!")

    except Exception as e:
        raise CustomException(e, sys)

def overthrow_the_champion(client, model_name, candidate, champion):
    try:
        client.set_model_version_tag(
        name=model_name,
        version=candidate.version,
        key="deployment_status",
        value="champion"
        )
        logging.info("Candidate model promoted to champion")

        client.set_model_version_tag(
        name=model_name,
        version=champion.version,
        key="deployment_status",
        value="archived"
        )
        logging.info("Champion model demoted")

    except Exception as e:
        raise CustomException(e, sys)


def main():
    try:

        model_name = "watch_price_predictor"
        client = MlflowClient()

        candidate_versions = client.search_model_versions(f"name='{model_name}' and tags.deployment_status='staging'")
        champion_versions = client.search_model_versions(f"name='{model_name}' and tags.deployment_status='champion'")

        candidate = candidate_versions[0]

        if not champion_versions:
            logging.info("Champion doesn't exist..")
            become_the_first_champion(client, model_name, candidate)
        else:
            overthrow_the_champion(client, model_name, candidate, champion_versions[0])
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()