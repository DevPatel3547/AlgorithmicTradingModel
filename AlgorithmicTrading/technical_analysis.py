
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
# Technical Indicators Analysis Function without TA-Lib
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