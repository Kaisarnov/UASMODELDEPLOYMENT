from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post("/predict", json={
        "features": {
            "Poor": 1,
            "Standard": 0,
            "Good": 25
        }
    })

    assert response.status_code == 200
    assert "prediction" in response.json()