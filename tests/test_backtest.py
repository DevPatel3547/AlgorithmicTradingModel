import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AlgorithmicTrading"))

from backtest import (  # noqa: E402
    combine_signals,
    mean_reversion_signal,
    regression_model,
    walk_forward_backtest,
)


def ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1e6},
        index=idx,
    )


def test_mean_reversion_signal_buys_dips_and_sells_spikes():
    close = pd.Series([100.0] * 30 + [90.0] + [100.0] * 30 + [110.0])
    s = mean_reversion_signal(close, window=20, z_entry=1.0)
    assert s.iloc[30] == 1  # far below the rolling mean
    assert s.iloc[-1] == -1  # far above it
    assert (s.iloc[:19] == 0).all()  # no signal before the window fills


def test_combine_signals_needs_the_forecast_up_and_no_mean_reversion_objection():
    pred = np.array([0.01, 0.01, 0.01, -0.01, -0.01, -0.01])
    mr = np.array([1, 0, -1, 1, 0, -1])
    # long only when the forecast is up and the price is not stretched above its mean
    assert combine_signals(pred, mr).tolist() == [1, 1, 0, 0, 0, 0]


def test_walk_forward_trains_only_on_the_past():
    data = ohlc()
    seen = []

    def spy(x_train, y_train):
        seen.append(len(x_train))
        return regression_model(x_train, y_train)

    res = walk_forward_backtest(data, seq_length=20, n_folds=4, min_train=200, model_factory=spy)
    # Training sets grow fold by fold, and every fold's test block starts where its training ends.
    assert seen == sorted(seen) and len(seen) == 4
    for _, train_end, test_end in res.folds:
        assert train_end < test_end
    assert res.daily.index.min() == data.index[200]
    assert len(res.daily) == len(data) - 200 - 1


def test_no_lookahead_changing_the_future_does_not_change_past_decisions():
    data = ohlc()
    base = walk_forward_backtest(data, seq_length=20, n_folds=2, min_train=200, model_factory=regression_model)
    future = data.copy()
    future.iloc[320:, :4] *= 3.0  # rewrite everything from row 320 on
    alt = walk_forward_backtest(future, seq_length=20, n_folds=2, min_train=200, model_factory=regression_model)
    cutoff = data.index[319]
    # The first fold (rows 200..299) was trained and traded before row 320 existed.
    first = base.daily.index < data.index[299]
    pd.testing.assert_series_equal(base.daily.loc[first, "position"], alt.daily.loc[first, "position"])
    pd.testing.assert_series_equal(
        base.daily.loc[first, "predicted_close"], alt.daily.loc[first, "predicted_close"]
    )
    assert cutoff in base.daily.index


def test_positions_earn_the_next_day_return():
    res = walk_forward_backtest(ohlc(), seq_length=20, n_folds=2, min_train=200, model_factory=regression_model)
    d = res.daily
    np.testing.assert_allclose(d["strategy_return"], d["position"] * d["market_return"])
    s = res.summary()
    assert s["test_days"] == len(d) and 0 <= s["direction_accuracy"] <= 1


def test_rejects_too_little_history():
    with pytest.raises(ValueError):
        walk_forward_backtest(ohlc(50), seq_length=60, model_factory=regression_model)
