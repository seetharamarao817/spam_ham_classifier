import logging
import sys
import os
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.logger.logging import logging
from src.exceptions.exception import customexception
from src.utils.utils import save_object, evaluate_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from dataclasses import dataclass

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    experiment_name = "Model_Training_Experiment"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'GausianNB': GaussianNB(),
                'MLP': MLPClassifier(max_iter=300, activation='logistic'),
                'svm': svm.SVC()
            }

            with mlflow.start_run(run_name="Model Training") as run:
                mlflow.log_params({"Train-Test Split": "75-25"})

                model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
                mlflow.log_metrics(model_report)

                best_model_score = max(sorted(model_report.values()))
                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
                best_model = models[best_model_name]

                mlflow.log_params({"Best Model": best_model_name, "R2 Score": best_model_score})

                save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
                
                # Log run name to MLflow
                client = MlflowClient()
                client.set_tag(run.info.run_id, MLFLOW_RUN_NAME, "Model Training")

        except Exception as e:
            logging.info('Exception occurred at Model Training')
            raise customexception(e,sys)
