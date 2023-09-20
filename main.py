from data_preprocessing import load_data, preprocess_data
from exploratory_analysis import analyze_data
from model import train_model, evaluate_model

def main():
    data = load_data('C:\\Users\\kevin\\Downloads\\house-prices (1)\\train.csv')
    ...


    clean_data = preprocess_data(data)
    analyze_data(clean_data)
    model, X_test, y_test = train_model(clean_data)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
