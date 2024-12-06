import os
import sys
import pandas as pd
import boto3
from io import StringIO
from dotenv import load_dotenv

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.pathmanager import PathManager

def ingest_data():
    try:
        load_dotenv()

        path_manager = PathManager()

        s3 = boto3.client('s3')

        bucket_name = os.getenv("BUCKET_NAME")
        file_key = os.getenv("FILE_KEY")

        if not bucket_name or not file_key:
            raise ValueError("BUCKET_NAME or FILE_KEY environment variables are not set")

        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_data = response['Body'].read().decode('utf-8') 
        
        data = pd.read_csv(StringIO(csv_data))

        raw_data_path = path_manager.get_raw_data_path()
        data.to_csv(raw_data_path, index=False)

        logging.info(f"Raw data saved to {raw_data_path}")

        return data

    except Exception as e:
        logging.error(f"Error in data ingestion: {str(e)}")
        raise CustomException(e, sys)

def main():
    ingest_data()

if __name__ == "__main__":
    main()