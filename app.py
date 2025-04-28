from flask import Flask, request, render_template_string
import numpy as np
from src.data_preprocessing import load_the_data, data_preprocessing
from src.feature_engineering import features_addition
from src.model import model_training
from src.evaluate import model_evaluation
import pandas as pd

app = Flask(__name__)

# Load and prepare data, train or load model once at startup
df = load_the_data("data/creditcard.csv")
df = data_preprocessing(df)
df['Card_ID'] = np.random.randint(1, 1000, df.shape[0])
df = features_addition(df)
model, X_test, y_test = model_training(df)

# Evaluate model and capture metrics as string
import io
import sys
output = io.StringIO()
sys.stdout = output
model_evaluation(model, X_test, y_test)
sys.stdout = sys.__stdout__
evaluation_report = output.getvalue()

# Simple HTML template for the web app with improved UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f7f8;
            margin: 0;
            padding: 0;
        }
        .container {
            max-width: 600px;
            margin: 50px auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        form {
            margin-top: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
        }
        input[type="submit"] {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
        .result {
            background-color: #e9f7ef;
            border: 1px solid #a6d8a8;
            padding: 15px;
            border-radius: 4px;
            color: #2e7d32;
            margin-top: 20px;
            font-weight: bold;
            text-align: center;
        }
        .error {
            background-color: #fbe9e7;
            border: 1px solid #f44336;
            padding: 15px;
            border-radius: 4px;
            color: #c62828;
            margin-top: 20px;
            font-weight: bold;
            text-align: center;
        }
        h2 {
            color: #333;
            margin-top: 40px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        pre {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            max-height: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Credit Card Fraud Detection</h1>
        <form method="post" action="/">
            <label>Amount:</label>
            <input type="number" step="0.01" name="Amount" required>
            <label>Card ID (1-1000):</label>
            <input type="number" name="Card_ID" min="1" max="1000" required>
            <input type="submit" value="Predict Fraud">
        </form>
        {% if prediction is not none %}
            {% if 'Error' in prediction %}
                <div class="error">{{ prediction }}</div>
            {% else %}
                <div class="result">{{ prediction }}</div>
            {% endif %}
        {% endif %}
        <h2>Model Evaluation Metrics:</h2>
        <pre>{{ evaluation_report }}</pre>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            amount = float(request.form["Amount"])
            card_id = int(request.form["Card_ID"])
            # Create a single sample dataframe with required features
            sample = pd.DataFrame({
                "Amount": [amount],
                "Card_ID": [card_id]
            })
            # Add features as in features_addition
            sample['Transaction_Count'] = df[df['Card_ID'] == card_id].shape[0] + 1
            sample['Average_Spending'] = (df[df['Card_ID'] == card_id]['Amount'].mean() * df[df['Card_ID'] == card_id].shape[0] + amount) / sample['Transaction_Count']
            # Align columns with training data except 'Class'
            feature_cols = model.feature_names_in_
            sample = sample.reindex(columns=feature_cols, fill_value=0)
            pred = model.predict(sample)[0]
            prediction = "Fraudulent Transaction" if pred == 1 else "Legitimate Transaction"
        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"
    return render_template_string(HTML_TEMPLATE, prediction=prediction, evaluation_report=evaluation_report)

if __name__ == "__main__":
    app.run(debug=True)
