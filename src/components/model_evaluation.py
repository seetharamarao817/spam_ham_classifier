import logging
import sys
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.logger import logging
from src.exception import CustomException
from src.utils.utils import save_object

class ModelEvaluator:
    @staticmethod
    def evaluate_model(X_train, y_train, X_test, y_test, models, run):
        try:
            report = {}
            for model_name, model_instance in models.items():
                model_run_name = f"Evaluation - {model_name}"
                mlflow.start_run(run_name=model_run_name, nested=True)
                model_instance.fit(X_train, y_train)
                y_test_pred = model_instance.predict(X_test)
                model_accuracy = accuracy_score(y_test, y_test_pred)
                classification_rep = classification_report(y_test, y_test_pred)
                confusion_mat = confusion_matrix(y_test, y_test_pred)
                logging.info(f"Classification report of {model_name} : {classification_report}")
                # Logging to MLflow using the provided run object
                mlflow.log_params({"Model Name": model_name})
                mlflow.log_metrics({"Accuracy": model_accuracy})

                report[model_name] = model_accuracy

                return report
        except Exception as e:
            logging.info('Exception occurred during model evaluation')
            raise CustomException(e, sys)
        finally:
            # End the nested MLflow run even if an exception occurs
            mlflow.end_run()
