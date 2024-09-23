from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.MentalHealthAnalysis.pipeline.prediction import PredictionPipeline
import joblib

app = Flask(__name__)

# Prediction pipeline object
obj = PredictionPipeline()

@app.route('/', methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])  # route to train the pipeline
def training():
    try:
        os.system("python main.py")
        return "Training Successful!"
    except Exception as e:
        print(f'Training failed: {e}')
        return f"Training failed: {e}"

@app.route('/predict', methods=['POST'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            # Get user input from form
            Thoughts = str(request.form['Thoughts'])

            # Preprocess input to match training data structure
            input_data = pd.DataFrame({
                'index': [1],  # Add a dummy 'index' column
                'statement': [Thoughts],
                'num_of_characters': [len(Thoughts)],  # Example: Number of characters
                'num_of_sentences': [Thoughts.count('.')],  # Example: Number of sentences
                'tokens': [len(Thoughts.split())],  # Example: Number of tokens
                'tokens_stemmed': [len(Thoughts.split())]  # Example: Number of stemmed tokens
            })

            # Debugging: Print columns to verify they match the expected format
            print(f"Input data columns: {input_data.columns.tolist()}")

            # Ensure no issues with index
            input_data.reset_index(drop=True, inplace=True)  # Ensure index isn't part of the columns

            # Load the model
            model_pipeline = joblib.load('artifacts/model_trainer/model.joblib')

            # Make prediction using the PredictionPipeline object
            predict = model_pipeline.predict(input_data)

            # Render the prediction result on the results page
            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print(f'The Exception message is: {e}')
            return f'Something went wrong: {e}'

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
