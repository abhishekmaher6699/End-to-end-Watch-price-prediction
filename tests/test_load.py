import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import os

from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

@pytest.mark.parametrize("model_name, deployment_status", [
    ("watch_price_predictor", "staging"),])
def test_load_latest_staging_model(model_name, deployment_status):
    
    client = MlflowClient()
    
    all_versions = client.search_model_versions(f"name='{model_name}'")
    filtered_versions = [
        version for version in all_versions
        if version.tags.get("deployment_status") == deployment_status
    ]
    latest_version = max(filtered_versions, key=lambda v: int(v.version), default=None)
    
    assert latest_version is not None, f"No model found in the '{deployment_status}' for '{model_name}'"

    try:
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version.version} loaded successfully from '{deployment_status}' stage.")

    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")