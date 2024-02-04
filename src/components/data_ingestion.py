import pandas as pd
import logging
import sys

class DataIngestion:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    def read_dataset(self)-> DataFrame:
        try:
            logging.info("Reading the dataset")
            self.dataset = pd.read_csv(self.file_path, sep='\t', names=['labels', 'message'])
            return self.dataset

        except Exception as e:
            logging.error("Error occurred in data ingestion part")
            raise CustomException(e, sys)

    def display_dataset_info(self):
        try:
            if self.dataset is not None:
                logging.info("Dataset shape: {}".format(self.dataset.shape))
                logging.info("Null values in the dataset:\n{}".format(self.dataset.isna().sum()))
            else:
                logging.warning("Dataset not loaded. Call read_dataset() method first.")
        except Exception as e:
            logging.error("Error occurred while displaying dataset information")
            raise CustomException(e, sys)

# Define your custom exception class
class CustomException(Exception):
    def __init__(self, message, source):
        super().__init__(message)
        self.source = source

# Example usage:
try:
    data_ingestion = DataIngestion(r"C:\Users\DELL\Downloads\sms+spam+collection\SMSSpamCollection")
    data_ingestion.read_dataset()
    data_ingestion.display_dataset_info()
except CustomException as ce:
    print("Custom Exception: {}".format(ce))
except Exception as e:
    print("Unexpected error: {}".format(e))
