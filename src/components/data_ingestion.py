import pandas as pd
from src.logger import logging
import sys
from src.exception import CustomException

class DataIngestion:

    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None

    def read_dataset(self):
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


