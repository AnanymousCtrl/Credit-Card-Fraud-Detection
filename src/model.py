from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

def model_training(df):
    model_path = "saved_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # We still need to split data to get X_test, y_test for evaluation
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)
        return model, X_test, y_test
    else:
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        return model, X_test, y_test
