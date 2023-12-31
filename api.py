"""
Description:
    This is our business forcasting api application
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from pred_data_processing import data_formating
import numpy as np
import os

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    """
    Description:
        This function is the local host 1313 main route, where our render_template contains
        information about or api call
    """
    # RENDER TEMPLATE
    return render_template('index.html')

# Route for prediction using the GLM model with 0.75 prediction
@app.route('/predict', methods=['POST'])
def predict():
    """
    Description:
        This funtion is the main api route for obtaining our predictions
    """
    try:
        # Assuming the input data is in JSON format
        input_data = request.json

        if not input_data:
            # If there is no input data, return an empty list
            return jsonify([])

        # Make predictions using the loaded model
        with open('glm_model.pickle', 'rb') as file:
            glm_model = pickle.load(file)

        # Format input data using data_formating
        predicted_outcome = data_formating(input_data)

        # Calculate Predictions (Unit test length is > 1 for probs)
        predicted_outcome['business_outcome'] = glm_model.predict(predicted_outcome)

        # Calculate Probabilities
        """
        Note: If the data is one entry, it will return nan.
        Was not sure if to fix this by implemening the already know phat
        """
        predicted_outcome['phat'] = pd.qcut(predicted_outcome['business_outcome'], q=[0, .25, .5, .75, 1.], duplicates='drop').astype(str)

        # Filter prediction for 75th percentile & arrange in alphabetical order
        predicted_outcome = predicted_outcome.reindex(sorted(predicted_outcome.columns), axis=1)
        predicted_outcome = predicted_outcome[predicted_outcome['business_outcome'] >= 0.75]

        # Convert DataFrame to a list of dictionaries
        result_list = predicted_outcome.to_dict(orient='records')

        # Return the predictions as json
        return jsonify(result_list)

    except Exception as e:
        return jsonify({'error': str(e)})

# Route for prediction using the GLM model with 0.75 prediction
@app.route('/predict_all', methods=['POST'])
def predict_all():
    """
    Description:
        This funtion is the main api route for obtaining our predictions
    """
    try:
        # Assuming the input data is in JSON format
        input_data = request.json

        if not input_data:
            # If there is no input data, return an empty list
            return jsonify([])

        # Make predictions using the loaded model
        with open('glm_model.pickle', 'rb') as file:
            glm_model = pickle.load(file)

        # Format input data using data_formating
        predicted_outcome = data_formating(input_data)

        # Calculate Predictions (Unit test length is > 1 for probs)
        predicted_outcome['business_outcome'] = glm_model.predict(predicted_outcome)

        # Calculate Probabilities
        """
        Note: If the data is one entry, it will return nan.
        Was not sure if to fix this by implemening the already know phat
        """
        predicted_outcome['phat'] = pd.qcut(predicted_outcome['business_outcome'], q=[0, .25, .5, .75, 1.], duplicates='drop').astype(str)

        # Filter prediction for 75th percentile & arrange in alphabetical order
        predicted_outcome = predicted_outcome.reindex(sorted(predicted_outcome.columns), axis=1)

        # Convert DataFrame to a list of dictionaries
        result_list = predicted_outcome.to_dict(orient='records')

        # Return the predictions as json
        return jsonify(result_list)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1313)
