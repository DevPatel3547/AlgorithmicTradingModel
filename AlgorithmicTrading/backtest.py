"""Walk-forward backtest of the LSTM forecast combined with a mean-reversion signal.

Every fold trains only on history that ends before the fold starts: the scaler is
fitted on the training rows alone, the model never sees a test row, and each daily
decision uses only data available at that day's close. The position taken at the
close of day t earns the return from t to t+1.

Transaction costs and slippage are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["Open", "High", "Low", "Close"]
CLOSE = FEATURES.index("Close")

# A model factory takes (x_train, y_train) and returns a predict(x) -> y callable.
# x has shape (n, seq_length, n_features), y has shape (n,), both scaled to [0, 1].
ModelFactory = Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]


def mean_reversion_signal(close: pd.Series, window: int = 20, z_entry: float = 1.0) -> pd.Series:
    """+1 when the close is z_entry standard deviations below its rolling mean (expect a
    bounce up), -1 when it is that far above (expect a pull back), 0 otherwise.

    Uses a trailing window only, so the value on day t depends on days <= t.
    """
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    z = (close - mean) / std.replace(0, np.nan)
    signal = pd.Series(0, index=close.index, dtype=int)
    signal[z < -z_entry] = 1
    signal[z > z_entry] = -1
    return signal


def combine_signals(predicted_return: np.ndarray, mr_signal: np.ndarray) -> np.ndarray:
    """Long-only position from the two signals.

    The forecast votes +1 (predicted up) or -1 (predicted down); the mean-reversion
    signal votes +1, 0 or -1. Go long only when the combined vote is at least +1: the
    forecast must predict a rise, and the price must not be stretched above its mean.
    A mean-reversion buy alone cannot outvote a forecast of a fall.
    """
    vote = np.sign(predicted_return) + np.asarray(mr_signal)
    return (vote >= 1).astype(int)


def _windows(scaled: np.ndarray, seq_length: int, ends: range) -> np.ndarray:
    """Sequences of seq_length rows ending at (and including) each index in ends."""
    return np.stack([scaled[e - seq_length + 1 : e + 1] for e in ends])


def lstm_model(x_train: np.ndarray, y_train: np.ndarray):
    """The Keras LSTM from model_building.py (TensorFlow is imported only when used)."""
    from model_building import create_model

    model = create_model((x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=0)
    return lambda x: model.predict(x, verbose=0).reshape(-1)


def regression_model(x_train: np.ndarray, y_train: np.ndarray):
    """Linear regression on the flattened window: a fast baseline for the LSTM."""
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x_train.reshape(len(x_train), -1), y_train)
    return lambda x: reg.predict(x.reshape(len(x), -1))


@dataclass
class BacktestResult:
    daily: pd.DataFrame  # one row per test day: close, predicted_close, mr, position, returns
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    @property
    def strategy_return(self) -> float:
        return float((1 + self.daily["strategy_return"]).prod() - 1)

    @property
    def buy_and_hold_return(self) -> float:
        return float((1 + self.daily["market_return"]).prod() - 1)

    @property
    def direction_accuracy(self) -> float:
        d = self.daily
        return float((np.sign(d["predicted_return"]) == np.sign(d["market_return"])).mean())

    @property
    def sharpe(self) -> float:
        r = self.daily["strategy_return"]
        return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0

    def summary(self) -> dict:
        return {
            "test_days": len(self.daily),
            "folds": len(self.folds),
            "strategy_return": self.strategy_return,
            "buy_and_hold_return": self.buy_and_hold_return,
            "direction_accuracy": self.direction_accuracy,
            "days_in_market": int(self.daily["position"].sum()),
            "sharpe": self.sharpe,
        }


def walk_forward_backtest(
    data: pd.DataFrame,
    seq_length: int = 60,
    n_folds: int = 4,
    min_train: int | None = None,
    model_factory: ModelFactory = lstm_model,
    mr_window: int = 20,
    mr_z: float = 1.0,
) -> BacktestResult:
    """Train on everything before each fold, trade the fold, then roll forward.

    The history after the first min_train rows is cut into n_folds consecutive test
    blocks. For each block the scaler and model are fitted on rows [0, start) only.
    """
    data = data[FEATURES].dropna()
    n = len(data)
    min_train = min_train if min_train is not None else n // 2
    if min_train <= seq_length + 1:
        raise ValueError("min_train must leave room for at least one training sequence")
    if n - min_train < n_folds + 1:
        raise ValueError("not enough rows after min_train for the requested folds")

    values = data.to_numpy(dtype=float)
    close = data["Close"]
    mr = mean_reversion_signal(close, mr_window, mr_z).to_numpy()
    # The last row has no next-day return to trade, so test blocks stop at n - 2.
    bounds = np.linspace(min_train, n - 1, n_folds + 1).astype(int)

    rows = []
    folds = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        scaler = MinMaxScaler().fit(values[:start])  # training rows only
        scaled = scaler.transform(values)
        # Training pairs: window ending at e -> close at e + 1, with e + 1 < start.
        train_ends = range(seq_length - 1, start - 1)
        x_train = _windows(scaled, seq_length, train_ends)
        y_train = scaled[[e + 1 for e in train_ends], CLOSE]
        predict = model_factory(x_train, y_train)

        test_ends = range(start, end)  # decide at close of day e, earn e -> e + 1
        pred_scaled = np.asarray(predict(_windows(scaled, seq_length, test_ends)), dtype=float)
        lo, hi = scaler.data_min_[CLOSE], scaler.data_max_[CLOSE]
        pred_close = pred_scaled * (hi - lo) + lo
        for k, e in enumerate(test_ends):
            today, tomorrow = values[e, CLOSE], values[e + 1, CLOSE]
            rows.append(
                {
                    "date": data.index[e],
                    "close": today,
                    "predicted_close": pred_close[k],
                    "predicted_return": pred_close[k] / today - 1,
                    "mr_signal": int(mr[e]),
                    "market_return": tomorrow / today - 1,
                }
            )
        folds.append((data.index[0], data.index[start], data.index[end]))

    daily = pd.DataFrame(rows).set_index("date")
    daily["position"] = combine_signals(daily["predicted_return"].to_numpy(), daily["mr_signal"].to_numpy())
    daily["strategy_return"] = daily["position"] * daily["market_return"]
    return BacktestResult(daily=daily, folds=folds)
