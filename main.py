import pandas as pd
import data_preprocessing as dp
import exploratory_analysis as ea
import model as ml

# Cargar y preprocesar datos de entrenamiento
data = dp.load_data('https://raw.githubusercontent.com/kevinbarra/machine-learning/main/train.csv')
data = dp.fill_na_values(data)
data = dp.remove_outliers(data)
data.reset_index(drop=True, inplace=True)  # Restablecer índices

# Análisis exploratorio y selección de características
ea.plot_correlation(data)
features = ea.select_features(data, "SalePrice")

# Definir las características para la predicción (sin la columna 'SalePrice')
features_for_prediction = [f for f in features if f != "SalePrice"]

# Entrenar y evaluar el modelo
model, X_test, y_test = ml.train_model(data, features, "SalePrice")
mse = ml.evaluate_model(model, X_test, y_test)
print(f"Mean Squared Error: {mse}")

# Cargar y preprocesar datos de prueba
test_data = dp.load_data('https://raw.githubusercontent.com/kevinbarra/machine-learning/main/test.csv')
test_data = dp.fill_na_values(test_data)
test_data = dp.remove_outliers(test_data)
test_data.reset_index(drop=True, inplace=True)  # Restablecer índices

# Predecir precios para el conjunto de prueba
predicted_prices = model.predict(test_data[features_for_prediction])
predictions_df = pd.DataFrame({
    'Id': test_data['Id'],
    'Predicted_SalePrice': predicted_prices
})

# Guardar las predicciones en un archivo CSV
predictions_df.to_csv('predicted_prices.csv', index=False)

print(f"Predicciones guardadas en 'predicted_prices.csv'")

# Predecir precios para el conjunto de entrenamiento y comparar
train_predictions = model.predict(data[features_for_prediction])
comparison_df = pd.DataFrame({
    'Id': data['Id'],
    'Actual_SalePrice': data['SalePrice'],
    'Predicted_SalePrice': train_predictions
})

# Guardar las comparaciones en un archivo CSV
comparison_df.to_csv('train_price_comparison.csv', index=False)

print(f"Comparaciones guardadas en 'train_price_comparison.csv'")

print(features)

