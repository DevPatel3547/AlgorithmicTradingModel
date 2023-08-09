import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold

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

# Create and compile the model
def create_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
    return model

# Train the model
def train_model(x_train, y_train):
    model = create_model((x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, batch_size=32, epochs=25)
    return model

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

# Make predictions
def make_predictions(model, x_test, scaler):
    prediction = model.predict(x_test)
    prediction_unscaled = np.concatenate((x_test[:, -1, :], prediction), axis=1)
    prediction_unscaled[:, -1] = prediction[:, 0]
    prediction = scaler.inverse_transform(prediction_unscaled)[:, -1]
    return prediction

# Plot the predictions
def plot_prediction(data, prediction, stock, seq_length):
    plt.figure(figsize=(15,6))
    plt.plot(data['Close'].values, color='blue', label='Actual Stock Price')
    plt.plot(range(seq_length, len(prediction) + seq_length), prediction, color='red', label='Predicted Stock Price')
    plt.title('Stock Price Prediction for ' + stock)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Function to predict future prices and provide buy/sell recommendations
def future_price_predictions(model, scaler, last_data, seq_length, forecast_length=7):
    input_data = last_data[-seq_length:, :4].copy()  # Including only 'Open', 'High', 'Low', 'Close'
    future_prices = []
    # Predict future prices
    for _ in range(forecast_length):
        scaled_input = scaler.transform(input_data[-seq_length:])
        scaled_input = scaled_input[:, :-1].reshape((1, seq_length, 3))
        predicted_price = model.predict(scaled_input)[0][0]
        input_data = np.vstack([input_data, [predicted_price, 0, 0, 0]])
        unscaled_prediction = scaler.inverse_transform(
            np.append(scaled_input[0, -1, :], predicted_price).reshape(1, -1)
        )[-1, -1]
        future_prices.append(unscaled_prediction)



    # Determine best days to buy/sell
    buy_days = []
    sell_day = 0
    max_profit = 0

    for i in range(forecast_length):
        for j in range(i + 1, forecast_length):
            profit = future_prices[j] - future_prices[i]
            if profit > max_profit:
                max_profit = profit
                buy_days = [i]
                sell_day = j
            elif profit == max_profit:
                buy_days.append(i)

    # Create a table to display the results
    result_table = pd.DataFrame({
        'Day': [f"Day {i + 1}" for i in range(forecast_length)],
        'Predicted Price': future_prices,
        'Action': ['Buy' if i in buy_days else 'Sell' if i == sell_day else '' for i in range(forecast_length)]
    })

    return result_table

# Moving Averages Analysis Function
def moving_averages(data, window_sizes=[5, 10, 20, 50]):
    ma_data = data.copy()
    for window in window_sizes:
        ma_data['MA' + str(window)] = data['Close'].rolling(window=window).mean()
    return ma_data

# Volume Analysis Function
def plot_volume(data):
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    ax1.plot(data['Close'], color='blue')
    ax2.bar(data.index, data['Volume'], color='gray')
    ax1.set_title('Price and Volume')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='blue')
    ax2.set_ylabel('Volume', color='gray')
    plt.show()

# Technical Indicators Analysis Function
def compute_technical_indicators(data):
    ti_data = data.copy()

    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    ti_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    ti_data['MACD'] = exp1 - exp2

    # Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    ti_data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    ti_data['Bollinger_Middle'] = rolling_mean
    ti_data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)

    return ti_data


# Function to Plot Technical Indicators
def plot_technical_indicators(ti_data):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(15, 12))
    ax1.plot(ti_data['Close'], label='Close Price')
    ax1.set_title('Technical Indicators')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    ax2.plot(ti_data['RSI'], label='RSI')
    ax2.set_ylabel('RSI')
    ax2.legend(loc='best')
    ax3.plot(ti_data['MACD'], label='MACD')
    ax3.set_ylabel('MACD')
    ax3.legend(loc='best')
    ax4.plot(ti_data['Bollinger_Upper'], label='Bollinger Upper')
    ax4.plot(ti_data['Bollinger_Middle'], label='Bollinger Middle')
    ax4.plot(ti_data['Bollinger_Lower'], label='Bollinger Lower')
    ax4.set_ylabel('Bollinger Bands')
    ax4.legend(loc='best')
    plt.show()
    
    
    
    
def backtest_model(data, seq_length, window_size='1M'):
    cumulative_profit = 0
    profits = []
    holding_stock = False
    buy_price = 0
    
    # Resampling the data by the window size (e.g., monthly)
    resampled_data = data.resample(window_size).agg({'Open': 'first', 
                                                    'High': 'max', 
                                                    'Low': 'min', 
                                                    'Close': 'last',
                                                    'Volume': 'sum'})
    
    for i in range(seq_length + 41, len(resampled_data) - 1):
        # Training Data

        scaled_data, x_train, y_train, scaler = preprocess_data(data, seq_length)
        model = train_model(x_train, y_train)
        ti_data = compute_technical_indicators(data)  # Assuming you have a function for this

        # Testing Data (One window forward)
        test_data = resampled_data.iloc[i:i+1]
        actual_price = test_data['Close'].iloc[0]
        rsi_value = ti_data['RSI'].iloc[i - (seq_length + 42)] # Getting the corresponding RSI value

        # Apply Buy/Sell strategy
        stop_loss_threshold = 0.05 # 5% loss
        take_profit_threshold = 0.05 # 5% profit

        # Buy condition: RSI below 30 and not holding stock
        if rsi_value < 30 and not holding_stock:
            buy_price = actual_price
            holding_stock = True

        # Sell condition: RSI above 70 and holding stock
        elif holding_stock and rsi_value > 70:
            profit = actual_price - buy_price
            cumulative_profit += profit
            profits.append(cumulative_profit)
            holding_stock = False
            
    # Reporting
    if profits and profits[-1] > 0: # Making sure there's profit to report
        plt.plot(profits)
        plt.title('Backtesting Cumulative Profits')
        plt.xlabel('Window')
        plt.ylabel('Profit')
        plt.show()

    return profits




   

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
    
    profits = backtest_model(data, seq_length)
    print("Backtesting Completed. Cumulative Profits:", profits)
    

if __name__ == "__main__":
    main()
