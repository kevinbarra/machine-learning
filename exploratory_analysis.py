def analyze_data(data):
    # Solo selecciona columnas numéricas
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    
    # Calculamos la correlación con 'SalePrice'
    correlation = numeric_data.corr()
    print(correlation['SalePrice'].sort_values(ascending=False))

    # Aquí puedes agregar más análisis y visualizaciones si lo deseas

    # Aquí puedes agregar más análisis y visualizaciones si lo deseas
