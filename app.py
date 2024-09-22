from flask import Flask, render_template, request
import os
import numpy as np
from src.MentalHealthAnalysis.pipeline.prediction import PredictionPipeline
#from src.MentalHealthAnalysis.utils.utilities import load_model, preprocess_statement  # Import utility functions

app = Flask(__name__)

# # Load the model from the utils
# model = load_model()

# # Prediction pipeline object
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

@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            Thoughts = str(request.form['Thoughts'])

            data = np.array([Thoughts]).reshape(1, -1)

            # Make prediction using the model
            #prediction = model.predict(data)

            # Prepare the response
            predict = obj.predict(data)
            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print(f'The Exception message is: {e}')
            return f'Something went wrong: {e}'

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
