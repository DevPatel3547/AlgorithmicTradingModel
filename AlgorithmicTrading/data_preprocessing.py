import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler







# Function to get stock symbol from user
def get_user_stock():
    stock_symbol = input("Please enter the stock symbol (e.g., AAPL for Apple Inc.): ").upper().strip()
    return stock_symbol

# Function to load stock data
def load_data(stock, start_date, end_date):
    return yf.download(stock, start=start_date, end=end_date)[['Open', 'High', 'Low', 'Close', 'Volume']]

# Function to preprocess data for training
def preprocess_data(data, seq_length):
    # Scaling only the required columns
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close']])
    x_train, y_train = [], []
    for i in range(seq_length, len(scaled_data)):
        x_train.append(scaled_data[i-seq_length:i, :-1])
        y_train.append(scaled_data[i, -1])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))
    
    return scaled_data, x_train, y_train, scaler

# Create test data
def create_test_data(scaled_data, scaler, seq_length):
    test_data_unscaled = scaled_data[len(scaled_data) - seq_length:]
    test_data = scaler.transform(test_data_unscaled)
    x_test, y_test = [], []
    for i in range(seq_length, len(test_data)):
        x_test.append(test_data[i-seq_length:i, :-1])
        y_test.append(test_data[i, -1])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))
    y_test = np.array(y_test)
    return x_test, y_test

