from src.logger import logging
import sys
from src.exception import CustomException
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.utils import load_object, load_tfidf_vectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

class PredictionPipeline:
    def __init__(self,input,model_path="/workspaces/spam_ham_classifier/artifacts/model.pkl",vectorizer_path="/workspaces/spam_ham_classifier/artifacts/tfidf_vectorizer.pkl"):
        self.input = input
        self.model_path = model_path
        self.corpus = []
        self.X = None
        self.model = None
        self.vectorizer_path = vectorizer_path
        
    def run_pipeline(self):
        try:
            # Data Preprocessing
            self.input_preprocessing()
            self.transform_data()
        
            self.load_model()
            
            # Prediction
            predict = self.predict()
            return predict
        except Exception as ce:
            raise CustomException(ce, sys)


    def input_preprocessing(self):
        try:
            logging.info("Applying stemming and lemmatization")
            stemmer = PorterStemmer()
            lemmatizer = WordNetLemmatizer()

            
            text = re.sub('[^a-zA-Z1-9]', ' ', self.input)
            text = text.lower()
            tokens = nltk.word_tokenize(text)
            tokens = [lemmatizer.lemmatize(j) for j in tokens if j not in set(stopwords.words('english'))]
            text = ' '.join(tokens)
            self.corpus.append(text)
            

            logging.info("input preprocessing completed")
        except Exception as e:
            raise CustomException(e,sys)

    def transform_data(self):
        try:
            logging.info("Applying Term Frequency-Inverse Document Frequency (TF-IDF) transformation.")
            tfidf = load_tfidf_vectorizer(self.vectorizer_path)
            self.X = tfidf.transform(self.corpus).toarray()
        
        except Exception as e:
            raise CustomException(e, sys)

    def load_model(self):
        try:
            logging.info("Loading Model")
            self.model = load_object(self.model_path)
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self):
        try:
            logging.info("Making Predictions")
            predictions = self.model.predict(self.X)
            logging.info("Predictions: {}".format(predictions))
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

# Example usage:
try:
    input = "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update "
    prediction_pipeline = PredictionPipeline(input, r"/workspaces/spam_ham_classifier/artifacts/model.pkl","/workspaces/spam_ham_classifier/artifacts/tfidf_vectorizer.pkl")
    prediction_pipeline.run_pipeline()
except Exception as e:
    raise CustomException(e,sys)



