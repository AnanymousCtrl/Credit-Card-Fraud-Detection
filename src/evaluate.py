from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def model_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy Score:\n", accuracy_score(y_test, y_pred))

