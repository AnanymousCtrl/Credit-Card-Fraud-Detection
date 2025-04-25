import pandas as pd

def features_addition(df):
    df['Transaction_Count'] = df.groupby('Card_ID')['Amount'].transform('count')
    df['Average_Spending'] = df.groupby('Card_ID')['Amount'].transform('mean')
    return df
    