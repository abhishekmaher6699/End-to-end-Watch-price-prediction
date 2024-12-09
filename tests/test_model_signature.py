import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import pickle
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

@pytest.fixture
def columns():
    column_file_path = os.path.abspath("artifacts/models/columns.pkl")
    with open(column_file_path, 'rb') as f:
        return pickle.load(f)
    
@pytest.mark.parametrize("model_name, deployment_status", [
    ("watch_price_predictor", "staging"),])
def test_model(model_name, deployment_status, columns):
    client = MlflowClient()
    
    all_versions = client.search_model_versions(f"name='{model_name}'")
    filtered_versions = [
        version for version in all_versions
        if version.tags.get("deployment_status") == deployment_status
    ]
    assert filtered_versions, f"No versions found for model '{model_name}' with status '{deployment_status}'"
    
    latest_version = max(filtered_versions, key=lambda v: int(v.version), default=None)
    assert latest_version, "Unable to determine the latest model version"

    columns = columns

    try:
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)

        test_input = pd.DataFrame([['Maurice Lacroix', 'Automatic', 39.0, 11.0, 'Round', 'Steel','Sapphire Crystal', 'Steel', 'Folding', 'Men', 
                                        200.0, 2.0, 'Switzerland', 38.0, 26.0, 0, 'No Bezel', 0, 28800.0, 'No coating',
                                        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]], columns=columns)
        
        prediction = model.predict(test_input)
        print(prediction[0])
        assert prediction is not None, "Prediction is None"
        assert isinstance(prediction[0], (float, int)), f"Unexpected prediction type: {type(prediction[0])}"

    except Exception as e:
        pytest.fail(f"Model prediction failed with error: {e}")

        
