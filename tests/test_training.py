import numpy as np
import pandas as pd

from fraud.features import FEATURES, add_features
from fraud.train import pick_threshold, time_split


def test_features_have_the_right_columns_and_values():
    df = pd.DataFrame([{**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": 3600 * 25, "Amount": 0.0}])
    out = add_features(df)
    assert list(out.columns) == FEATURES
    assert out["hour"].iloc[0] == 1          # 25 hours after the start = 1 AM next day
    assert out["log_amount"].iloc[0] == 0.0  # log(1 + 0) = 0


def test_time_split_keeps_order_and_sizes():
    df = pd.DataFrame({"Time": np.arange(100)[::-1], "Class": 0})  # given in reverse order
    train, val, test = time_split(df)
    assert (len(train), len(val), len(test)) == (64, 16, 20)
    assert train["Time"].max() < val["Time"].min() < test["Time"].min()  # no future data in training


def test_threshold_moves_lower_when_misses_cost_more():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.9])
    cheap_misses = pick_threshold(y, proba, cost_miss=0.5)
    costly_misses = pick_threshold(y, proba, cost_miss=10)
    assert costly_misses <= cheap_misses
    # With costly misses we must catch the fraud scored 0.4.
    assert costly_misses <= 0.4
