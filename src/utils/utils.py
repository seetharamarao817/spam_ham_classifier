import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
def load_tfidf_vectorizer(path):
        try:
            tfidf_vectorizer = joblib.load(path)
            return tfidf_vectorizer
        except Exception as e:
            raise CustomException(e, sys)

    