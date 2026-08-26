from src.agent import respond_to_query
from src.tools import execute_tool, run_calculator, search_knowledge


def test_calculator_tool_runs_safe_math():
    result = execute_tool("calculator", {"expression": "2 + 3 * 4"})
    assert result["ok"] is True
    assert "14" in result["result"]


def test_knowledge_search_returns_content_for_basic_plan():
    result = search_knowledge("Basic Plan pricing")
    assert result["ok"] is True
    assert "Basic Plan" in result["result"]


def test_agent_handles_travel_request_differently():
    response = respond_to_query("make a trip from bengaluru to kerala")
    lower = response.lower()
    assert "trip" in lower or "itineraries" in lower or "AutoStream" in lower
    assert "not set up to plan trips" in lower or "itineraries" in lower
