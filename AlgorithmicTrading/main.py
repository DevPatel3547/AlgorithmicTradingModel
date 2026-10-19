
import pandas as pd
from data_preprocessing import get_user_stock, load_data, preprocess_data, create_test_data
from model_building import train_model
from predictions import make_predictions, plot_prediction, future_price_predictions
from technical_analysis import moving_averages, plot_volume, compute_technical_indicators, plot_technical_indicators
import yfinance as yf # For the Volume data
import matplotlib.pyplot as plt
from backtest import walk_forward_backtest, lstm_model

def main():
    stock = get_user_stock()
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    seq_length = 60

    data = load_data(stock, start_date, end_date)
    data['Volume'] = yf.download(stock, start=start_date, end=end_date)['Volume']
    scaled_data, x_train, y_train, scaler = preprocess_data(data, seq_length)

    # Walk-forward backtest on held-out history: each fold trains only on the days
    # before it, then trades the LSTM forecast combined with a mean-reversion signal.
    result = walk_forward_backtest(data, seq_length=seq_length, n_folds=4, model_factory=lstm_model)
    print("Walk-forward backtest (no transaction costs):")
    for key, value in result.summary().items():
        print(f"  {key}: {value}")

    # Final model for the forward-looking forecast below, trained on all history.
    model = train_model(x_train, y_train)


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