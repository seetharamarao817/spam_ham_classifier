import streamlit as st
from src.logger import logging
import sys
from src.exception import CustomException
from pipelines.prediction_pipeline import PredictionPipeline  #



def main():
    st.title("SPAM HAM Classifier")

    # Get user input
    user_input = st.text_input("Enter text to classify:")

    
    

    # Button to trigger classification
    if st.button("Classify"):
        try:
            # Run prediction pipeline
            prediction_pipeline = PredictionPipeline(user_input)
            prediction = prediction_pipeline.run_pipeline()
            if prediction == True:
                st.write("This message is classified as a SPAM")
            elif prediction == False:
                st.write("This message is classified as a HAM")
        except CustomException as e:
            st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
