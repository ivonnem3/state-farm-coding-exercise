"""
Description:
    This python file is primarly used to test our api calls.
"""
import unittest
import json
from flask import Flask
from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Set up the Flask app for testing
        app.testing = True
        self.app = app.test_client()

    def test_single_entry_request(self):
        """
         Description:
            Test Case: API can handle a JSON input request with one entry
        """

        # Load sample input for a single entry from a JSON file
        with open("sample_single_entry.json", "r") as file:
            sample_input = json.load(file)

        # Make a POST request to the /predict endpoint
        response = self.app.post('/predict', json=sample_input)

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Parse the response JSON
        results = response.json

        # Validate the structure of the response
        self.validate_response_structure(results)

    def test_multiple_entries_request(self):
        """
         Description:
            Test Case: API can handle a JSON input request with multiple entries
        """

        # Load sample input for multiple entries from a JSON file
        with open("sample_multiple_entries.json", "r") as file:
            sample_input = json.load(file)

        # Make a POST request to the /predict endpoint
        response = self.app.post('/predict', json=sample_input)

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Parse the response JSON
        results = response.json

        # Validate the structure of the response
        self.validate_response_structure(results)

    def test_incomplete_request(self):
        """
         Description:
            Test Case: API can handle an incomplete JSON input request with missing values
        """

        # Load sample incomplete input from a JSON file
        with open("sample_incomplete_entry.json", "r") as file:
            sample_input = json.load(file)

        # Make a POST request to the /predict endpoint
        response = self.app.post('/predict', json=sample_input)

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Parse the response JSON
        results = response.json

        # Validate the structure of the response
        self.validate_response_structure(results)

    def test_empty_request(self):
        """
         Description:
            Test Case: API can handle an empty JSON input entry
        """

        # Make a POST request to the /predict endpoint with an empty JSON
        response = self.app.post('/predict', json={})

        # Check if the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)

        # Parse the response JSON
        results = response.json

        # Ensure that the response is an empty list
        self.assertEqual(results, [])

    def validate_response_structure(self, results):
        # Validate the structure of each result in the response
        if isinstance(results, list):  # Check if the response is a list (successful case)
            for result in results:
                # Check if the required keys are present in the response
                self.assertIn('business_outcome', result)
                self.assertIn('phat', result)
        elif isinstance(results, dict) and 'error' in results:  # Check if the response is an error
            # Handle error response, you can choose to print or log the error
            print(f"Error response: {results['error']}")
        else:
            # Unexpected response format
            self.fail(f"Unexpected response format: {results}")

if __name__ == '__main__':
    unittest.main()
