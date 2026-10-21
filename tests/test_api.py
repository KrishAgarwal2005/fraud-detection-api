from tests.conftest import make_transaction


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_between_0_and_1(client):
    response = client.post("/predict", json=make_transaction())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction_id"] == 1


def test_high_risk_transaction_is_flagged(client):
    # The test model learned "V1 > 1 means fraud", so a large V1 should be flagged.
    assert client.post("/predict", json=make_transaction(v1=5.0)).json()["is_fraud"] is True
    assert client.post("/predict", json=make_transaction(v1=-5.0)).json()["is_fraud"] is False


def test_prediction_is_saved_and_can_be_read_back(client):
    new_id = client.post("/predict", json=make_transaction(amount=123.45)).json()["prediction_id"]
    saved = client.get(f"/predictions/{new_id}").json()
    assert saved["amount"] == 123.45
    assert saved["hour"] == 2          # time = 2 hours after the start


def test_unknown_prediction_returns_404(client):
    assert client.get("/predictions/999").status_code == 404


def test_wrong_number_of_features_is_rejected(client):
    bad = make_transaction()
    bad["v"] = [0.0] * 27              # one feature missing
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_amount_is_rejected(client):
    assert client.post("/predict", json=make_transaction(amount=-1)).status_code == 422


def test_list_can_filter_to_fraud_only(client):
    client.post("/predict", json=make_transaction(v1=5.0))
    client.post("/predict", json=make_transaction(v1=-5.0))
    flagged = client.get("/predictions", params={"only_fraud": True}).json()
    assert len(flagged) == 1
    assert flagged[0]["is_fraud"] == 1


def test_stats_count_requests_and_flags(client):
    client.post("/predict", json=make_transaction(v1=5.0))
    client.post("/predict", json=make_transaction(v1=-5.0))
    stats = client.get("/stats").json()
    assert stats["total_predictions"] == 2
    assert stats["flagged_as_fraud"] == 1
    assert stats["flag_rate"] == 0.5
