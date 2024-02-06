from src.logger import logging
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.exception.py import CustomException
from src.componets import ModelTrainer

class TrainingPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset = None
        self.corpus = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self):
        try:
            # Start MLflow run
            with mlflow.start_run(run_name="Training Pipeline") as run:
                # Data Ingestion
                self.data_ingestion()
                
                # Data Preprocessing
                self.data_preprocessing()
                
                # Data Transformation
                self.data_transformation()
                
                # Model Training
                self.model_training()

                # Log run name to MLflow
                client = MlflowClient()
                client.set_tag(run.info.run_id, MLFLOW_RUN_NAME, "Training Pipeline")

        except CustomException as ce:
            print("Custom Exception: {}".format(ce))
        except Exception as e:
            print("Unexpected error: {}".format(e))

    def data_ingestion(self):
        try:
            logging.info("Data Ingestion")
            data_ingestion = DataIngestion(self.file_path)
            self.dataset = data_ingestion.read_dataset()
            data_ingestion.display_dataset_info()
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise CustomException(e, sys)

    def data_preprocessing(self):
        try:
            logging.info("Data Preprocessing")
            data_preprocessing = DataPreprocessing(self.dataset)
            data_preprocessing.separate_features()
            self.corpus = data_preprocessing.apply_stemming_and_lemmatization()
            logging.info("Length of the corpus after preprocessing: {}".format(self.corpus))
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise CustomException(e, sys)

    def data_transformation(self):
        try:
            logging.info("Data Transformation")
            data_transformer = DataTransformer(self.corpus, self.y)
            data_transformer.transform_data()
            self.X_train = data_transformer.X_train
            self.X_test = data_transformer.X_test
            self.y_train = data_transformer.y_train
            self.y_test = data_transformer.y_test
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise CustomException(e, sys)

    def model_training(self):
        try:
            logging.info("Model Training")
            self.model_trainer.initate_model_training(
                np.column_stack((self.X_train, self.y_train)),
                np.column_stack((self.X_test, self.y_test))
            )
        except CustomException as ce:
            raise ce
        except Exception as e:
            raise CustomException(e, sys)

# Example usage:
try:
    pipeline = TrainingPipeline(r"C:\Users\DELL\Downloads\sms+spam+collection\SMSSpamCollection")
    pipeline.run_pipeline()
except CustomException as ce:
    print("Custom Exception: {}".format(ce))
except Exception as e:
    print("Unexpected error: {}".format(e))