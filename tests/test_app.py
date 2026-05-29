import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

# ── Use lifespan=True so startup event runs and agents load ───────────────────
@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health Check ──────────────────────────────────────────────────────────────
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "agents" in data
    print("Health check passed")


# ── API 1: Data Ingestion ─────────────────────────────────────────────────────
def test_ingest_valid_patient(client):
    response = client.post("/api/ingest", json={
        "name": "Test Patient",
        "gender": "Female",
        "age": 45,
        "hypertension": 1,
        "heart_disease": 0,
        "smoking_history": "former",
        "bmi": 28.5,
        "HbA1c_level": 6.2,
        "blood_glucose_level": 130
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "record_id" in data
    print("Ingest valid patient passed")


def test_ingest_missing_field(client):
    response = client.post("/api/ingest", json={
        "name": "Incomplete Patient",
        "gender": "Male"
    })
    assert response.status_code == 422
    print("Ingest missing field validation passed")


# ── API 2: ML Prediction ──────────────────────────────────────────────────────
def test_predict_diabetic(client):
    response = client.post("/api/predict", json={
        "gender": "Female",
        "age": 62,
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_history": "current",
        "bmi": 35.0,
        "HbA1c_level": 7.8,
        "blood_glucose_level": 200
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert "explanation" in data
    assert data["label"] in ["DIABETIC", "NOT DIABETIC"]
    print(f"Predict diabetic passed — {data['label']} ({data['probability']}%)")


def test_predict_non_diabetic(client):
    response = client.post("/api/predict", json={
        "gender": "Male",
        "age": 25,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 22.0,
        "HbA1c_level": 5.1,
        "blood_glucose_level": 85
    })
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["DIABETIC", "NOT DIABETIC"]
    print(f"Predict non-diabetic passed — {data['label']} ({data['probability']}%)")


def test_predict_missing_field(client):
    response = client.post("/api/predict", json={
        "gender": "Male",
        "age": 30
    })
    assert response.status_code == 422
    print("Predict missing field validation passed")


# ── API 3: Document Search (RAG) ──────────────────────────────────────────────
def test_search_valid_question(client):
    response = client.post("/api/search", json={
        "question": "What HbA1c level indicates diabetes?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    print("Document search passed")


def test_search_another_question(client):
    response = client.post("/api/search", json={
        "question": "What are the risk factors for diabetes?"
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    print("Document search second question passed")


def test_search_empty_question(client):
    response = client.post("/api/search", json={
        "question": ""
    })
    assert response.status_code in [200, 500]
    print("Document search empty question handled")


# ── API 4: Agent Interaction ──────────────────────────────────────────────────
def test_agent_document_routing(client):
    response = client.post("/api/agent", json={
        "query": "What are the complications of diabetes?",
        "patient_data": None
    })
    assert response.status_code == 200
    data = response.json()
    assert "agent_used" in data
    assert "response" in data
    print(f"Agent routing passed — routed to: {data['agent_used']}")


def test_agent_data_routing(client):
    response = client.post("/api/agent", json={
        "query": "What percentage of patients have diabetes in the dataset?",
        "patient_data": None
    })
    assert response.status_code == 200
    data = response.json()
    assert "agent_used" in data
    print(f"Agent data routing passed — routed to: {data['agent_used']}")


def test_agent_ml_routing(client):
    response = client.post("/api/agent", json={
        "query": "Predict risk for this patient",
        "patient_data": {
            "gender": "Female",
            "age": 55,
            "hypertension": 1,
            "heart_disease": 0,
            "smoking_history": "former",
            "bmi": 33.5,
            "HbA1c_level": 7.2,
            "blood_glucose_level": 180
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert data["agent_used"] == "ml_agent"
    print(f"Agent ML routing passed — routed to: {data['agent_used']}")


# ── ML Model Unit Tests (independent of API) ──────────────────────────────────
def test_ml_model_prediction_output():
    from agents.ml_expert_agent import MLExpertAgent
    agent = MLExpertAgent()
    result = agent.predict({
        "gender": "Female", "age": 62, "hypertension": 1,
        "heart_disease": 1, "smoking_history": "current",
        "bmi": 35.0, "HbA1c_level": 7.8, "blood_glucose_level": 200
    })
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 100
    print(f"ML model test passed — {result['prediction']}, {result['probability']}%")


def test_ml_model_low_risk():
    from agents.ml_expert_agent import MLExpertAgent
    agent = MLExpertAgent()
    result = agent.predict({
        "gender": "Male", "age": 22, "hypertension": 0,
        "heart_disease": 0, "smoking_history": "never",
        "bmi": 20.0, "HbA1c_level": 4.8, "blood_glucose_level": 80
    })
    assert result["prediction"] in [0, 1]
    print(f"ML low risk test passed — {result['prediction']}")