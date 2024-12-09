import pandas as pd
import numpy as np

import sys
import os
import yaml
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate
import mlflow
import mlflow.sklearn

from catboost import CatBoostRegressor

from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.pathmanager import PathManager

from tqdm.auto import tqdm

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Params loaded successfully!")
        return params
    except Exception as e:
        raise CustomException(e, sys)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded from the file")
        return df
    except Exception as e:
        raise CustomException(e, sys)
    
def split_data(df):

    try:
        y_train = np.log1p(df['price'])
        X_train = df.drop('price', axis=1)
        logging.info("Train data ready for model")

        return X_train, y_train

    except Exception as e:
        raise CustomException(e, sys)
    

def create_pipeline(X_train, params):
    try:

        numerical_cols = ['Case Size', 'Case Thickness', 'Power Reserve', 'Frequency (bph)', 'Jewels', 'Water Resistance (M)', 'Warranty Period']

        binary_cat_cols = [
            col for col in X_train.columns
            if X_train[col].nunique() == 2
        ]

        multi_cat_cols = X_train.drop(columns=binary_cat_cols + numerical_cols).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num_scaler', StandardScaler(), numerical_cols),
                ('binary_cols', 'passthrough', binary_cat_cols),
                ('cat_encoder', OneHotEncoder(handle_unknown='ignore'), multi_cat_cols)
            ])
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', CatBoostRegressor(**params))
        ])
        logging.info("Pipeline is ready")

        return pipeline
    
    except Exception as e:
        raise CustomException(e, sys)
    
def train_pipeline(pipeline, X_train, y_train):
    try:
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline trained succesfully")
        return pipeline
    except Exception as e:
        raise CustomException(e, sys)
    
def save_model(pipeline, columns, path_manager):
    try:
        
        with open(path_manager.get_pipeline_path(), 'wb') as f:
            pickle.dump(pipeline, f)
        logging.info("Pipeline saved!")

        with open(path_manager.get_columns_path(), 'wb') as f:
            pickle.dump(columns, f)
        logging.info("Columns saved!")
        
    except Exception as e:
        raise CustomException(e, sys)
    
def main():

    try:
        path_manager = PathManager()
        df = load_data(path_manager.get_train_data_path())
        params_yaml = load_params(path_manager.get_params_path())

        X_train, y_train = split_data(df)
        columns = X_train.columns

        params =  {
            'iterations': params_yaml['model_building']['iterations'],
            'learning_rate': params_yaml['model_building']['learning_rate'],
            'depth': params_yaml['model_building']['depth'],
            'l2_leaf_reg': params_yaml['model_building']['l2_leaf_reg'],
            'rsm': params_yaml['model_building']['rsm'],
            'random_strength': params_yaml['model_building']['random_strength'],
            'bagging_temperature': params_yaml['model_building']['bagging_temperature'],
            'border_count': params_yaml['model_building']['border_count'],
            'verbose': 0,
            'loss_function': params_yaml['model_building']['loss_function']
        }

        pipeline = create_pipeline(X_train, params)
        pipeline = train_pipeline(pipeline, X_train, y_train)
        save_model(pipeline, columns, path_manager)
    
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    main()