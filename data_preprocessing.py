import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath, index_col='Id')


def preprocess_data(data):
    # Reemplazar NaN en columnas numéricas con la mediana
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)

    # Reemplazar NaN en columnas categóricas con la moda
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Aquí puedes agregar más preprocesamiento si es necesario

    return data
