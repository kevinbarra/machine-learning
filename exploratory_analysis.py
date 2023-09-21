import pandas as pd
# exploratory_analysis.py

import matplotlib.pyplot as plt
import numpy as np

def plot_correlation(data):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()
    plt.figure(figsize=(12, 9))
    plt.matshow(correlation, fignum=1)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.colorbar()
    plt.show()

def select_features(data, target_column):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation = numeric_data.corr()[target_column]
    relevant_features = correlation[correlation.abs() > 0.5].index.tolist()
    return relevant_features

