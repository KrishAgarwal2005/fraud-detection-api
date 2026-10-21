# Credit Card Fraud Detection API

A fraud model trained on 284,807 real card transactions (only 0.17% are fraud), served through a FastAPI REST API that scores a transaction and stores every prediction in a SQLite database.

**Result on the held-out test period:** the chosen XGBoost model caught **57 of 75 frauds (76% recall) with 85% precision**, raising **10 false alarms out of 56,887 normal transactions**. Test PR-AUC is **0.79**.

## Results

Data: the public [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (European cardholders, two days in September 2013). Columns V1 to V28 are anonymised (the bank applied PCA), plus `Time`, `Amount` and `Class`.

Transactions are split **by time**: first 64% for training, next 16% for validation, last 20% for testing. The test set is only used once, at the end.

| Model | Val PR-AUC | Test PR-AUC | Threshold | Precision | Recall | F1 | Frauds caught | False alarms |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression (class weights) | 0.7513 | 0.7387 | 0.999 | 0.6548 | 0.7333 | 0.6918 | 55 of 75 | 29 |
| Logistic Regression (SMOTE) | 0.7681 | 0.7496 | 0.999 | 0.6629 | 0.7867 | 0.7195 | 59 of 75 | 30 |
| Random Forest (class weights) | 0.7627 | 0.8138 | 0.26 | 0.9194 | 0.76 | 0.8321 | 57 of 75 | 5 |
| **XGBoost (scale_pos_weight)** | **0.7748** | 0.7921 | 0.78 | 0.8507 | 0.76 | 0.8028 | 57 of 75 | 10 |

Full numbers are in `reports/metrics.json`.

## Decisions worth explaining

- **Why not accuracy?** Always answering "not fraud" is already 99.87% accurate on the test set and catches nothing. So the models are judged on precision, recall and PR-AUC.
- **Why PR-AUC?** It measures how well the model ranks frauds above normal transactions when frauds are rare. A random model scores about 0.0013 here (the fraud rate).
- **Handling the imbalance.** Class weights (logistic regression, random forest), `scale_pos_weight` (XGBoost), and SMOTE for one logistic regression. SMOTE runs inside the pipeline, so it only ever sees training data.
- **Choosing the threshold.** The threshold is picked on the validation set to minimise cost, assuming a missed fraud costs 10 false alarms (`--cost-miss 10`, configurable). That is why each model has its own threshold instead of 0.5.
- **Choosing the model.** The winner is picked by validation PR-AUC, not test PR-AUC. Random Forest did better on the test set (0.81 vs 0.79), but choosing it because of the test set would be cheating, since the test set must stay unseen.
- **Splitting by time** avoids training on transactions from after the ones being tested.

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Model name, version and threshold |
| POST | `/predict` | Score one transaction, save the result, return probability and flag |
| GET | `/predictions?limit=20&only_fraud=false` | Most recent predictions |
| GET | `/predictions/{id}` | One saved prediction (404 if missing) |
| GET | `/stats` | Total predictions, number flagged, flag rate, average probability |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"time": 7200, "amount": 49.99, "v": [0.1, -1.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0, -1.1, 1.2, 0.3, -1.4, 0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.07, 0.08]}'
```

The request is validated: exactly 28 `v` values, and `time` and `amount` must not be negative (otherwise the API returns 422). Interactive docs are at `/docs`.

Predictions are stored with plain SQL in a `predictions` table (id, time created, amount, hour, probability, flag, threshold, model version), indexed on creation time.

## Run it

```bash
pip install -r requirements.txt

# 1. Download creditcard.csv from Kaggle into data/
# 2. Train (about 2 minutes); writes models/model.joblib and reports/
python -m fraud.train --data data/creditcard.csv

# 3. Start the API
uvicorn fraud.api:app --reload

# Tests (no dataset needed)
pytest -q
```

A trained model is already included in `models/`, so step 2 is optional.

## Tests

12 tests. The API tests use a tiny model trained on random numbers and a temporary database, so they run in under a second without the dataset. They check: valid predictions, that a high-risk transaction is flagged, that predictions are saved and can be read back, 404 for unknown ids, rejection of bad input, filtering to frauds only, and the stats counts. Other tests check the features, the time split, and that the threshold moves lower when misses cost more.

## Project layout

```
fraud/features.py   shared feature preparation (log amount, hour of day)
fraud/train.py      split, train 4 models, pick threshold and best model, save
fraud/db.py         SQLite storage with plain SQL
fraud/api.py        FastAPI app (create_app takes model and database paths)
tests/              API and training tests
reports/            metrics from the last training run
models/             saved model
```

## Limits

- V1 to V28 are anonymised, so the model cannot be explained in business terms.
- The data covers only two days, from 2013.
- One request takes about 60 ms on a 2-core laptop-class machine. Building one row of features with pandas is the slow part, and it has not been optimised.
