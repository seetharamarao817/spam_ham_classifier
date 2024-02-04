import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logging import logging
from src.exceptions.exception import customexception

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            
            model_accuracy = accuracy_score(y_test,y_test_pred)
            classification_rep = classification_report(y_test, y_test_pred)
            logging.info(f"Classification Report of {model}:\n{classification_rep}")
            
            confusion_mat = confusion_matrix(y_test, y_test_pred)
            logging.info(f"Confusion Matrix of {model }:\n{confusion_mat}")

            logging.info("Model evaluation completed successfully.")

            report[list(models.keys())[i]] =  model_accuracy

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise customexception(e,sys)

    