from src.logger import logging
from src.exception import  CustomException
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class DataTransformer:
    def __init__(self, corpus, y, test_size=0.25):
        self.corpus = corpus
        self.y = y
        self.test_size = test_size
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tf_idf = None

    def transform_data(self):
        try:
            logging.info("Applying Term Frequency-Inverse Document Frequency (TF-IDF) transformation.")
            
            # TF-IDF transformation
            self.tf_idf = TfidfVectorizer()
            X = self.tf_idf.fit_transform(self.corpus).toarray()

            # Splitting the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, self.y, test_size=self.test_size,stratify=self.y)
            joblib.dump(self.tf_idf, 'artifacts/tfidf_vectorizer.pkl')
            logging.info("Data transformation completed successfully.")
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            logging.error("Error occurred in data transformation process.")
            raise CustomException(e, sys)


