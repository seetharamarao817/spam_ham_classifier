import logging
import sys
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.logger.logging import logging
from src.exceptions.exception import CustomException
from dataclasses import dataclass

@dataclass 
class PredictionPipelineConfig:
    model_path = os.path.join('artifacts', 'model.pkl')

class PredictionPipeline:
    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):
        try:
            logging.info("Loading the trained model")
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                client = MlflowClient()
                model_uri = f"runs:/{run_id}/{self.model_path}"
                self.model = mlflow.pyfunc.load_model(model_uri)

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
    prediction_pipeline = PredictionPipeline(PredictionPipelineConfig.model_path)
    prediction_pipeline.load_model()

    # Assuming 'data' is the input data for prediction
    predictions = prediction_pipeline.predict(data)

    # Log run name to MLflow
    client = MlflowClient()
    client.set_tag(run.info.run_id, MLFLOW_RUN_NAME, "Prediction Pipeline")

except CustomException as ce:
    print("Custom Exception: {}".format(ce))
except Exception as e:
    print("Unexpected error: {}".format(e))
