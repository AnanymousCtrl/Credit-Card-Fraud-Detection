# Credit Card Fraud Detection App

This project is a credit card fraud detection system implemented in Python using machine learning. It includes data preprocessing, feature engineering, model training, evaluation, and a Flask web application for interactive fraud prediction.

## Features

- Loads and preprocesses credit card transaction data
- Adds engineered features for improved model performance
- Trains a Random Forest classifier with SMOTE for class imbalance handling
- Evaluates the model with confusion matrix, classification report, and accuracy score
- Provides a Flask web app for users to input transaction details and get fraud predictions
- Displays model evaluation metrics on the web interface
- Saves and loads the trained model to improve app startup time
- Clean and user-friendly web UI

## Installation

1. Clone the repository
2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure the dataset `data/creditcard.csv` is present
4. Data set link - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Usage

Run the Flask app:
```
python app.py
```

Open your browser and go to `http://127.0.0.1:5000/` to access the fraud detection web app.

## Project Structure

- `app.py`: Flask web application integrating the fraud detection pipeline
- `main.py`: Script demonstrating the full pipeline execution
- `src/`: Contains modules for data preprocessing, feature engineering, model training, and evaluation
- `data/creditcard.csv`: Dataset used for training and testing

## Model Applications

This fraud detection model can be adapted for other anomaly detection tasks such as:

- Insurance fraud detection
- Loan application fraud
- E-commerce transaction fraud
- Healthcare billing fraud
- Identity theft detection
- Cybersecurity anomaly detection
- Anti-money laundering monitoring

## License

This project is provided as-is for educational purposes.
