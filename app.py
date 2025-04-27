from flask import Flask, request, render_template_string
import numpy as np
from src.data_preprocessing import load_the_data, data_preprocessing
from src.feature_engineering import features_addition
from src.model import model_training
from src.evaluate import model_evaluation
import pandas as pd

app = Flask(__name__)

# Load and prepare data, train model once at startup
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

# Simple HTML template for the web app
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
</head>
<body>
    <h1>Credit Card Fraud Detection</h1>
    <form method="post" action="/">
        <label>Amount:</label><br>
        <input type="number" step="0.01" name="Amount" required><br><br>
        <label>Card ID (1-1000):</label><br>
        <input type="number" name="Card_ID" min="1" max="1000" required><br><br>
        <input type="submit" value="Predict Fraud">
    </form>
    {% if prediction is not none %}
        <h2>Prediction Result:</h2>
        <p>{{ prediction }}</p>
    {% endif %}
    <h2>Model Evaluation Metrics:</h2>
    <pre>{{ evaluation_report }}</pre>
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
