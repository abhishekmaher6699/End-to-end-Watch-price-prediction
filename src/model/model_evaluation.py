import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml
import pickle
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import mlflow
import mlflow.sklearn
import json
from mlflow.models import infer_signature

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.pathmanager import PathManager
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def load_pipeline(pipeline_path: str):
    try:
        with open(pipeline_path, 'rb') as file:
            model = pickle.load(file)
        logging.info(f"Model loaded from {pipeline_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Params loaded successfully")
        return params
    except Exception as e:
        raise CustomException(e, sys)

def prepare_data(df):
    try:
        X_test = df.drop('price', axis=1)
        y_test = np.log1p(df['price'])

        return X_test, y_test
    except Exception as e:
        raise CustomException(e, sys)
    
def plot_chart(test, pred):
    path_manager = PathManager()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    residuals = test - pred

    ax1 = axes[0]
    ax1.scatter(test, pred, alpha=0.6, edgecolor='k')
    ax1.set_xlabel("True Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(f"CatBoost: True vs Predicted Values")

    ax2 = axes[1]
    ax2.scatter(test, residuals, alpha=0.6, edgecolor='k')
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("True Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title(f"CatBoost: Residuals vs True Values")

    plt.tight_layout()

    plt.savefig(path_manager.get_chart_path())
    logging.info(f"Chart saved successfully")
    plt.close()

def evaluate(X_test, y_test, pipeline):
    try:
        y_pred = pipeline.predict(X_test)

        y_test_og = np.expm1(y_test)
        y_pred_og = np.expm1(y_pred)

        mse = mean_squared_error(y_test_og, y_pred_og)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_og, y_pred_og)
        r2 = r2_score(y_test_og, y_pred_og)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        plot_chart(y_test_og, y_pred_og)
        logging.info("Evaluation done")
        return metrics

    except Exception as e:
        raise CustomException(e, sys)
    
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.info("Model info saved")
    except Exception as e:
        raise CustomException(e, sys)

def main():
    try:
        path_manager = PathManager()

        mlflow.set_experiment("pipeline-runs")

        with mlflow.start_run() as run:
            test_df = load_data(path_manager.get_test_data_path())
            pipeline = load_pipeline(path_manager.get_pipeline_path())
            params = load_params(path_manager.get_params_path())

            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.log_param("run", run.info.run_id)

            X_test, y_test = prepare_data(test_df)
            metrics = evaluate(X_test, y_test, pipeline)

            input_example = X_test.iloc[[1]]
            signature = infer_signature(input_example, pipeline.predict(input_example))  

            mlflow.sklearn.log_model(
                pipeline,
                "catboost_pipeline",
                signature=signature, 
                input_example=input_example 
            )
            logging.info("Model logged")

            model_path = "catboost_pipeline"
            save_model_info(run.info.run_id, model_path, 'experiment_info.json')

            # with open(path_manager.get_columns_path(), 'rb') as file:
            #     columns = pickle.load(file)
            # mlflow.log_artifact(columns)

            mlflow.log_metrics(metrics)

            mlflow.set_tag("model_type", "CatBoost")
            mlflow.set_tag("task", "Price Prediction")
            mlflow.set_tag("dataset", "Luxury Watches")


    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == '__main__':
    main()