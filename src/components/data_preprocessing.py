import pandas as pd
from src.logger import logging
from src.exception import CustomException
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
            return self.y
        except Exception as e:
            logging.error("Error occurred in data preprocessing part")
            raise CustomException(e, sys)

    def apply_stemming_and_lemmatization(self):
        try:
            logging.info("Applying stemming and lemmatization")
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            for i in range(len(self.messages)):
                text = re.sub('[^a-zA-Z0-9]', ' ', self.messages[i])
                text = text.lower()
                tokens = nltk.word_tokenize(text)
                tokens = [lemmatizer.lemmatize(j) for j in tokens if j not in set(stopwords.words('english'))]
                text = ' '.join(tokens)
                self.corpus.append(text)

            logging.info("Data preprocessing completed")
            return self.corpus
        except Exception as e:
            logging.error("Error occurred in data preprocessing part")
            raise CustomException(e, sys)