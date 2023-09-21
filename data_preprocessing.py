# data_preprocessing.py

import pandas as pd
import numpy as np

def load_data(url):
    return pd.read_csv(url)

def fill_na_values(data):
    numeric_data = data.select_dtypes(include=[np.number])
    non_numeric_data = data.select_dtypes(exclude=[np.number])

    numeric_data = numeric_data.fillna(numeric_data.mean())
    non_numeric_data = non_numeric_data.fillna(non_numeric_data.mode().iloc[0])

    return pd.concat([numeric_data, non_numeric_data], axis=1)


def remove_outliers(data):
    numeric_data = data.select_dtypes(include=[np.number])
    
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    
    filtered_data = numeric_data[~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    non_numeric_data = data.select_dtypes(exclude=[np.number])
    non_numeric_data = non_numeric_data.loc[filtered_data.index]
    
    return pd.concat([filtered_data, non_numeric_data], axis=1)

