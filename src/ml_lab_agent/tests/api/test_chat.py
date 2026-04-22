import pytest
from fastapi.testclient import TestClient

from ml_lab_agent.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_response_show_all(client):
    response = client.post("/chat", json={"message": "show"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body

    assert body["intent"] == "show"
    assert body["error"] is None
    assert isinstance(body["data"], list)


def test_chat_response_show_one_id(client):
    response = client.post("/chat", json={"message": "show run 1"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "show"
    assert body["data"]["run_id"] == "1"
    assert body["error"] is None
    assert isinstance(body["data"], dict)


def test_chat_response_compare_two(client):
    response = client.post("/chat", json={"message": "compare run 1 and run 2"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "compare"
    assert body["error"] is None
    assert isinstance(body["data"], dict)


def test_chat_response_unknown_intent(client):
    response = client.post("/chat", json={"message": "hello there"})

    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "unknown"
    assert body["data"] is None
    assert body["error"] is not None


def test_chat_response_summarize_compare_three(client):
    response = client.post("/chat", json={"message": "compare run 1, 2 and 3"})

    assert response.status_code == 200
    body = response.json()

    assert "message" in body
    assert body["intent"] == "compare"
    assert body["error"] == "Can only accept two unique runs to compare."
    assert body["data"] is None
