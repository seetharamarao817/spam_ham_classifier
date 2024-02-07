import logging
import sys
import mlflow
import os
from src.utils.utils import load_object
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

@dataclass 
class PredictionPipelineConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class PredictionPipeline:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        try:
            logging.info("Loading the trained model")
            self.model = load_object(self.model_path)
            

        except Exception as e:
            logging.error("Error occurred while loading the model")
            raise CustomException(e, sys)

    def predict(self, data):
        try:
            logging.info("Making predictions")
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)

# Example usage:
try:
    data = "You have won 1000 dollars click here "
    prediction_pipeline = PredictionPipeline(PredictionPipelineConfig.model_path)
    prediction_pipeline.load_model()
    
    # Assuming 'data' is the input data for prediction
    predictions = prediction_pipeline.predict(data)

    # Log run name to MLflow
    print(predictions)
except CustomException as ce:
    print("Custom Exception: {}".format(ce))
except Exception as e:
    print("Unexpected error: {}".format(e))
