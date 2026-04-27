"""
test_api_endpoints.py
=====================
Integration tests for the FastAPI HTTP layer defined in backend/app.py.

WHAT IS BEING TESTED
--------------------
The HTTP contract of each endpoint:
  - Correct status codes (200, 500, …)
  - Response body shape (fields, types)
  - Delegation to RAGSystem — the right methods are called with the right args
  - Edge cases: missing session_id, injected session_id, error propagation

HOW THESE TESTS WORK
---------------------
The `api_client` fixture (conftest.py) builds an inline FastAPI app with the
same routes as app.py but without the StaticFiles mount or ChromaDB init.
A `mock_rag_system` (also from conftest.py) is wired in as the backend.
All tests use Starlette's TestClient so they exercise real HTTP serialisation.

RUN
---
    uv run pytest backend/tests/test_api_endpoints.py -v
"""

import pytest


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    def test_returns_200_with_valid_request(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is RAG?"})
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "session_id" in body

    def test_answer_matches_rag_system_output(self, api_client, mock_rag_system):
        mock_rag_system.query.return_value = ("Exact answer text.", [])
        resp = api_client.post("/api/query", json={"query": "any question"})
        assert resp.json()["answer"] == "Exact answer text."

    def test_new_session_created_when_session_id_omitted(self, api_client, mock_rag_system):
        """When no session_id is sent, session_manager.create_session must be called."""
        api_client.post("/api/query", json={"query": "hello"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_provided_session_id_is_forwarded_to_rag_query(self, api_client, mock_rag_system):
        """When the client sends a session_id it must NOT be replaced."""
        api_client.post(
            "/api/query",
            json={"query": "follow-up", "session_id": "my-session-42"},
        )
        mock_rag_system.session_manager.create_session.assert_not_called()
        _, called_session = mock_rag_system.query.call_args[0]
        assert called_session == "my-session-42"

    def test_returned_session_id_matches_created_session(self, api_client, mock_rag_system):
        mock_rag_system.session_manager.create_session.return_value = "brand-new-sid"
        resp = api_client.post("/api/query", json={"query": "hi"})
        assert resp.json()["session_id"] == "brand-new-sid"

    def test_sources_list_shape(self, api_client, mock_rag_system):
        """Each source must have 'label'; 'url' is optional but must be present in schema."""
        mock_rag_system.query.return_value = (
            "answer",
            [
                {"label": "Course A - Lesson 3", "url": "https://example.com/3"},
                {"label": "Course B - Lesson 1", "url": None},
            ],
        )
        sources = api_client.post("/api/query", json={"query": "q"}).json()["sources"]
        assert len(sources) == 2
        assert sources[0]["label"] == "Course A - Lesson 3"
        assert sources[0]["url"] == "https://example.com/3"
        assert sources[1]["url"] is None

    def test_empty_sources_list_is_valid(self, api_client, mock_rag_system):
        mock_rag_system.query.return_value = ("answer", [])
        resp = api_client.post("/api/query", json={"query": "q"})
        assert resp.json()["sources"] == []

    def test_rag_exception_returns_500(self, api_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB offline")
        resp = api_client.post("/api/query", json={"query": "q"})
        assert resp.status_code == 500
        assert "DB offline" in resp.json()["detail"]

    def test_missing_query_field_returns_422(self, api_client):
        """FastAPI/Pydantic validation: missing required field → 422 Unprocessable Entity."""
        resp = api_client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_query_is_forwarded_to_rag_system(self, api_client, mock_rag_system):
        api_client.post("/api/query", json={"query": "specific question text"})
        called_query = mock_rag_system.query.call_args[0][0]
        assert "specific question text" in called_query


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    def test_returns_200(self, api_client):
        resp = api_client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_contains_total_courses_and_titles(self, api_client):
        body = api_client.get("/api/courses").json()
        assert "total_courses" in body
        assert "course_titles" in body

    def test_total_courses_matches_analytics(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 5,
            "course_titles": ["A", "B", "C", "D", "E"],
        }
        body = api_client.get("/api/courses").json()
        assert body["total_courses"] == 5
        assert len(body["course_titles"]) == 5

    def test_course_titles_are_strings(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["RAG Course", "MCP Course"],
        }
        titles = api_client.get("/api/courses").json()["course_titles"]
        assert all(isinstance(t, str) for t in titles)

    def test_analytics_exception_returns_500(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("analytics failure")
        resp = api_client.get("/api/courses")
        assert resp.status_code == 500
        assert "analytics failure" in resp.json()["detail"]

    def test_get_course_analytics_called_once(self, api_client, mock_rag_system):
        api_client.get("/api/courses")
        mock_rag_system.get_course_analytics.assert_called_once()


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:

    def test_returns_200(self, api_client):
        resp = api_client.delete("/api/session/some-id")
        assert resp.status_code == 200

    def test_response_body_is_cleared_status(self, api_client):
        body = api_client.delete("/api/session/some-id").json()
        assert body == {"status": "cleared"}

    def test_clear_session_called_with_correct_id(self, api_client, mock_rag_system):
        api_client.delete("/api/session/my-unique-sid")
        mock_rag_system.session_manager.clear_session.assert_called_once_with("my-unique-sid")
