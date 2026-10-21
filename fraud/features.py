"""Feature preparation, shared by training and by the API.

Keeping this in one place means the model sees exactly the same columns
during training and when it serves predictions.
"""

import numpy as np
import pandas as pd

# The dataset gives 28 anonymised columns V1..V28 (the bank already applied PCA),
# plus Time (seconds since the first transaction) and Amount.
V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# Final list of columns the model uses, in this exact order.
FEATURES = V_COLUMNS + ["log_amount", "hour"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new table with the model's input columns.

    - log_amount: amounts are very skewed (most are small, a few are huge),
      so log(1 + amount) puts them on a more even scale.
    - hour: hour of the day, from Time. Fraud is more common at night.
    """
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour"] = (out["Time"] // 3600) % 24
    return out[FEATURES]
