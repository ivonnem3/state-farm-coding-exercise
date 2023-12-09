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

app = Flask(__name__)

@app.route('/')
def home():
    # RENDER TEMPLATE
    return render_template('index.html')

# Route for prediction using the GLM model
@app.route('/process_json', methods=['POST'])
def process_json():
    try:
        # Get the uploaded file
        json_file = request.files['jsonFile']

        # Load JSON data from the file
        data = json_file.read()

        # Convert JSON to a DataFrame
        input_data = pd.read_json(data)

        # Make predictions using the loaded model
        glm_model = pickle.load(open('glm_model.pickle','rb'))

        # Calculate probabilities and format predictions (Unit test length is > 1 for probs)
        predicted_outcome = input_data
        predicted_outcome['business_outcome']= glm_model.predict(predicted_outcome)
        predicted_outcome['phat'] = pd.qcut(predicted_outcome['business_outcome'], q = [0, .25, .5, .75, 1.]).astype(str)

        # Return the predictions as JSON
        return predicted_outcome.to_json(orient='records')[1:-1].replace('},{', '} {')

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=8080)
