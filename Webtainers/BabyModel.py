from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# Define your neural network model class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Load the model function
def load_model():
    input_size = 2048  # Size of Morgan fingerprint
    hidden_size = 128
    output_size = 1
    loaded_model = Net(input_size, hidden_size, output_size)
    checkpoint = torch.load("model.pt", map_location=torch.device('cpu'))
    loaded_model.load_state_dict(checkpoint)
    loaded_model.eval()
    return loaded_model

# Load the model when the Docker container starts
model = load_model()

# Function to convert SMILES string to Morgan fingerprints
def smiles_to_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return fingerprint

# Define a function to convert a SMILES string to a tensor
def smiles_to_tensor(smiles):
    fingerprint = smiles_to_morgan_fingerprint(smiles)
    tensor = torch.Tensor(fingerprint).unsqueeze(0)
    return tensor

# API endpoint to receive JSON input and provide predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data from the request
        input_data = request.get_json()

        # Extract the SMILES string from the input data
        smiles_input = input_data.get('smiles')

        # Make predictions
        with torch.no_grad():
            input_tensor = smiles_to_tensor(smiles_input)
            output = model(input_tensor)
            prediction = output.item()

        # Create a JSON response with the prediction
        result = {'prediction': prediction}

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
