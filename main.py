"""
For the API, please return the:
    - predicted outcome (variable name is business_outcome)
    - predicted probability (variable name is phat)
    - all model inputs;
the variables should be returned in alphabetical order in the API return.
"""
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def hello_world():
    # RENDER TEMPLATE
    return 'Hello, World! This is your API.'

# Route for prediction using the GLM model
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the input data is in JSON format
        input_data = request.json

        # Perform any necessary preprocessing on input_data if needed

        # Make predictions using the loaded model
        glm_model = pickle.load(open('glm_model.pickle','rb'))
        #predictions = glm_model.predict([input_data])

        # Return the predictions as JSON
        return input_data #jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
