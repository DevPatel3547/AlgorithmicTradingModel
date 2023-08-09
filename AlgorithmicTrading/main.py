
import pandas as pd
from data_preprocessing import get_user_stock, load_data, preprocess_data, create_test_data
from model_building import train_model
from predictions import make_predictions, plot_prediction, future_price_predictions
from technical_analysis import moving_averages, plot_volume, compute_technical_indicators, plot_technical_indicators
import yfinance as yf # For the Volume data
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

def main():
    stock = get_user_stock()
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    seq_length = 60

    data = load_data(stock, start_date, end_date)
    data['Volume'] = yf.download(stock, start=start_date, end=end_date)['Volume']
    scaled_data, x_train, y_train, scaler = preprocess_data(data, seq_length)

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=2, shuffle=True)
    for train_index, test_index in kfold.split(x_train):
        train_X, test_X = x_train[train_index], x_train[test_index]
        train_y, test_y = y_train[train_index], y_train[test_index]
        model = train_model(train_X, train_y)
        prediction = make_predictions(model, test_X, scaler)
        mse = mean_squared_error(test_y, prediction)
        print(f"Mean Squared Error for fold: {mse}")
        
    ma_data = moving_averages(data[['Close']])
    plt.figure(figsize=(15, 6))
    plt.plot(ma_data['Close'], label='Close Price')
    for col in ma_data.columns[1:]:
        plt.plot(ma_data[col], label=col)
    plt.title('Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Volume Analysis
    plot_volume(data)

    # Technical Indicators Analysis
    ti_data = compute_technical_indicators(data[['Close']])
    plot_technical_indicators(ti_data)

    # Future Price Predictions and Buy/Sell Recommendations
    recommendations = future_price_predictions(model, scaler, data.values, seq_length)
    print("Future Price Predictions and Buy/Sell Recommendations:")
    print(recommendations)
    
    

if __name__ == "__main__":
    main()