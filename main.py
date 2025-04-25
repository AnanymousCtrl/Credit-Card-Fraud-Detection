from src.data_preprocessing import load_the_data, data_preprocessing
from src.feature_engineering import features_addition
from src.model import model_training
from src.evaluate import model_evaluation

if __name__ == "__main__":
    df = load_the_data("data/creditcard.csv")
    df = data_preprocessing(df)

    import numpy as np
    df['Card_ID'] = np.random.randint(1,1000, df.shape[0])

    df =  features_addition(df)

    model, X_test, y_test =  model_training(df)
    model_evaluation(model, X_test, y_test)
    