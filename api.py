"""
For the API, please return the:
    - predicted outcome (variable name is business_outcome)
    - predicted probability (variable name is phat)
    - all model inputs;
the variables should be returned in alphabetical order in the API return.
"""
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from data_segmentation_processing import prediction_data_processing
import numpy as np
import os

app = Flask(__name__)
app.static_folder = 'static'

@app.route('/')
def home():
    # RENDER TEMPLATE
    return render_template('index.html')

# Route for prediction using the GLM model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the input data is in JSON format
        input_data = request.json

        # Perform any necessary preprocessing on input_data if needed
        input_data = pd.DataFrame(input_data).replace(np.nan,0)

        # Make predictions using the loaded model
        glm_model = pickle.load(open('glm_model.pickle','rb'))

        # Imprement sata data_segmentation_processing on the input data
        predicted_outcome = input_data#prediction_data_processing(input_data)

        #return str(predicted_outcome)

        # Calculate probabilities and format predictions (Unit test length is > 1 for probs)
        predicted_outcome['business_outcome']= glm_model.predict(predicted_outcome)
        predicted_outcome['phat'] = pd.qcut(predicted_outcome['business_outcome'], q=[0, .25, .5, .75, 1.], duplicates='drop').astype(str)

        # Filter prediction for 75th percentile & arrange in alphabetical order
        #predicted_outcome = predicted_outcome[predicted_outcome['business_outcome'] >= 0.75]
        #predicted_outcome = predicted_outcome.reindex(sorted(predicted_outcome.columns), axis=1)

        # Convert DataFrame to a list of dictionaries
        result_list = predicted_outcome.to_dict(orient='records')

        # Return the predictions as json
        return jsonify(result_list)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=1313)
