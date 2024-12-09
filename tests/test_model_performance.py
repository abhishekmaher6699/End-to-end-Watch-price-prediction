import mlflow.pyfunc
import pytest
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from dotenv import load_dotenv

# load_dotenv()

# import dagshub
# dagshub.init(repo_owner='abhishekmaher6699', repo_name='End-to-end-Watch-price-prediction', mlflow=True)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

@pytest.fixture
def validation_data():
    df = pd.read_csv("artifacts/data/processed/test.csv")
    X = df.drop('price', axis=1)
    Y = np.log1p(df['price'])
    return X, Y

@pytest.mark.parametrize("model_name", [
    ("watch_price_predictor"), 
])
def test_model_performance(model_name, validation_data):

    X, y = validation_data

    client = MlflowClient()

    candidate_versions = client.search_model_versions(f"name='{model_name}' and tags.deployment_status='staging'")
    champion_versions = client.search_model_versions(f"name='{model_name}' and tags.deployment_status='champion'")

    assert candidate_versions, "No staging model found for deployment."
    candidate = candidate_versions[0]

    if not champion_versions:
        assert candidate, "No champion model found for comparison."
        
        candidate_model_uri = f"models:/{model_name}/{candidate.version}"
        candidate_model = mlflow.pyfunc.load_model(candidate_model_uri)

        y_pred = candidate_model.predict(X)
        y_og = np.expm1(y) 
        y_pred_og = np.expm1(y_pred)

        mse = mean_squared_error(y_og, y_pred_og)
        candidate_rmse = np.sqrt(mse)
        candidate_mae = mean_absolute_error(y_og, y_pred_og)
        candidate_r2 = r2_score(y_og, y_pred_og)

        assert candidate_rmse < 500000, f"RMSE of candidate model is too high: {candidate_rmse}"
        assert candidate_mae < 200000, f"MAE of candidate model is too high: {candidate_mae}"
        assert candidate_r2 > 0.85, f"R2 score of candidate model is too low: {candidate_r2}"
        print(f"Candidate model meets the threshold with RMSE: {candidate_rmse}, MAE: {candidate_mae}, R2: {candidate_r2}")

    else:
        champion = champion_versions[0]

        candidate_model_uri = f"models:/{model_name}/{candidate.version}"
        champion_model_uri = f"models:/{model_name}/{champion.version}"

        candidate_model = mlflow.pyfunc.load_model(candidate_model_uri)
        champion_model = mlflow.pyfunc.load_model(champion_model_uri)

        y_pred_candidate = candidate_model.predict(X)
        y_pred_champion = champion_model.predict(X)

        y_og = np.expm1(y)
        y_pred_candidate_og = np.expm1(y_pred_candidate)
        y_pred_champion_og = np.expm1(y_pred_champion)

        mse_candidate = mean_squared_error(y_og, y_pred_candidate_og)
        candidate_rmse = np.sqrt(mse_candidate)
        candidate_mae = mean_absolute_error(y_og, y_pred_candidate_og)
        candidate_r2 = r2_score(y_og, y_pred_candidate_og)

        mse_champion = mean_squared_error(y_og, y_pred_champion_og)
        champion_rmse = np.sqrt(mse_champion)
        champion_mae = mean_absolute_error(y_og, y_pred_champion_og)
        champion_r2 = r2_score(y_og, y_pred_champion_og)

        assert candidate_rmse < champion_rmse, f"Candidate RMSE is higher than the champion RMSE: {candidate_rmse} > {champion_rmse}"
        assert candidate_mae < champion_mae, f"Candidate MAE is higher than the champion MAE: {candidate_mae} > {champion_mae}"
        assert candidate_r2 > champion_r2, f"Candidate R2 is lower than the champion R2: {candidate_r2} < {champion_r2}"

        print(f"Candidate model RMSE: {candidate_rmse}, MAE: {candidate_mae}, R2: {candidate_r2}")
        print(f"Champion model RMSE: {champion_rmse}, MAE: {champion_mae}, R2: {champion_r2}")
