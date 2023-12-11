import unittest
import requests
import json

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Start the Flask app in testing mode
        from api import app
        app.testing = True
        self.app = app.test_client()

    def test_predict_endpoint(self):
        # Test Case: Proper code execution
         # Load sample input from a JSON file
        with open("sample_input.json", "r") as file:
            sample_input = json.load(file)

        # Make a POST request to the /predict endpoint
        response = self.app.post('/predict', json=sample_input)

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Parse the response JSON
        results = response.json

        # Iterate over each result in the list and check for keys
        for result in results:
            # Check if the required keys are present in the response
            self.assertIn('business_outcome', result)
            self.assertIn('phat', result)

            # Print the result for inspection
            print(result)

if __name__ == '__main__':
    unittest.main()
