import numpy as np
import matplotlib.pyplot as plt
import pandas as pd






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

