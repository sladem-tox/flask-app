from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Render the input form page
@app.route('/') # represents last part of URL eg /Model1 etc.
def input_form():
    return render_template('input_form.html')

# API endpoint to process user input and get model predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the user input from the form
        smiles_input = request.form['smiles']

        # Prepare JSON data to send to the model container
        input_data = {'smiles': smiles_input}

        # Send a POST request to the model container API
        model_url = 'http://model-container:5000/predict'  # Assumes the model container is named 'model-container'
        response = requests.post(model_url, json=input_data)

        if response.status_code == 200:
            # Successful response from the model container
            result = response.json()
            prediction = result.get('prediction')

            # Display the prediction result
            return render_template('prediction_result.html', prediction=prediction)
        else:
            return render_template('error.html', error='Model container error')

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
