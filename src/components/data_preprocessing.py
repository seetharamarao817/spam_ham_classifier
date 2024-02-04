import pandas as pd
import logging
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

class DataPreprocessing:

    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = None
        self.messages = None
        self.y = None
        self.corpus = []

    def separate_features(self):
        try:
            logging.info("Separating independent and dependent features")
            self.labels = self.dataset['labels']
            self.messages = self.dataset['message']
            self.y = pd.get_dummies(self.labels).iloc[:, 1].values
        except Exception as e:
            logging.error("Error occurred in data preprocessing part")
            raise CustomException(e, sys)

    def apply_stemming_and_lemmatization(self):
        try:
            logging.info("Applying stemming and lemmatization")
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            for i in range(len(self.messages)):
                text = re.sub('[^a-zA-Z1-9]', ' ', self.messages[i])
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                tokens = [lemmatizer.lemmatize(j) for j in tokens if j not in set(stopwords.words('English'))]
                text = ' '.join(tokens)
                self.corpus.append(text)

            logging.info("Data preprocessing completed")
            return len(self.corpus)
        except Exception as e:
            logging.error("Error occurred in data preprocessing part")
            raise CustomException(e, sys)

# Define your custom exception class
class CustomException(Exception):
    def __init__(self, message, source):
        super().__init__(message)
        self.source = source

# Example usage:
try:
    data_preprocessing = DataPreprocessing(ds)
    data_preprocessing.separate_features()
    corpus_length = data_preprocessing.apply_stemming_and_lemmatization()
    logging.info("Length of the corpus after preprocessing: {}".format(corpus_length))
except CustomException as ce:
    print("Custom Exception: {}".format(ce))
except Exception as e:
    print("Unexpected error: {}".format(e))
