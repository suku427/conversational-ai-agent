import os

from fastapi.testclient import TestClient

from main import app
from src.rag import retrieve_context


def test_retrieve_context_for_basic_plan():
    results = retrieve_context("What is the price of the Basic Plan?", limit=3)
    assert results
    assert any("Basic Plan" in item["content"] for item in results)


def test_chat_api_returns_response():
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={"prompt": "What is the price of the Basic Plan?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "response" in payload
    assert any(keyword in payload["response"].lower() for keyword in ["basic plan", "29", "price"])
