import logging
import sys
import mlflow
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.logger.logging import logging
from src.exceptions.exception import customexception
from src.utils.utils import save_object

class ModelEvaluator:
    @staticmethod
    def evaluate_model(X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            for model_name, model_instance in models.items():
                with mlflow.start_run(run_name=f"Evaluation - {model_name}") as run:
                    model_instance.fit(X_train, y_train)
                    y_test_pred = model_instance.predict(X_test)
                    model_accuracy = accuracy_score(y_test, y_test_pred)
                    classification_rep = classification_report(y_test, y_test_pred)
                    confusion_mat = confusion_matrix(y_test, y_test_pred)

                    # Logging to MLflow
                    mlflow.log_params({"Model Name": model_name})
                    mlflow.log_metrics({"Accuracy": model_accuracy})
                    mlflow.log_artifact("classification_report.txt", classification_rep)
                    mlflow.log_artifact("confusion_matrix.txt", confusion_mat)

                    report[model_name] = model_accuracy

                    # Log run name to MLflow
                    client = MlflowClient()
                    client.set_tag(run.info.run_id, MLFLOW_RUN_NAME, f"Evaluation - {model_name}")

            return report

        except Exception as e:
            logging.info('Exception occurred during model evaluation')
            raise customexception(e, sys)
