from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def select_features(data):
    # Seleccionamos solo las columnas numéricas
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Calculamos la correlación con 'SalePrice' para las columnas numéricas
    correlation = numeric_data.corr()['SalePrice'].sort_values(ascending=False)
    
    # Seleccionamos las características basadas en tu recomendación y otras con alta correlación
    # Aquí puedes ajustar el número de características según lo que consideres adecuado
    top_features = correlation.index[1:6].tolist()  # Tomamos las 5 características con mayor correlación (sin contar 'SalePrice')
    
    return top_features

def train_model(data):
    features = select_features(data)
    X = data[features]
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio: {mse}")
