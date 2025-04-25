import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_the_data(url):
    df = pd.read_csv(url)
    return df

def data_preprocessing(df):
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    df.drop(['Time'], axis =1, inplace=True)
    return df