"""
conftest.py — shared fixtures loaded automatically before any test run.

The backend/ directory is already on sys.path via pyproject.toml's
[tool.pytest.ini_options] pythonpath = ["backend"], so no manual
sys.path manipulation is needed here.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Shared RAGSystem mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_rag_system():
    """
    A fake RAGSystem wired with sensible defaults.

    Tests that need different return values can override them directly:
        mock_rag_system.query.return_value = ("other answer", [])
    """
    rag = MagicMock()
    rag.session_manager.create_session.return_value = "test-session-abc"
    rag.query.return_value = (
        "Here is the answer.",
        [{"label": "RAG Course - Lesson 1", "url": "https://example.com/lesson1"}],
    )
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["RAG Course", "MCP Course"],
    }
    return rag


# ---------------------------------------------------------------------------
# TestClient fixture for API endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture
def api_client(mock_rag_system):
    """
    TestClient wired to an inline FastAPI app that mirrors app.py's routes.

    We define the app here rather than importing backend/app.py because app.py
    mounts StaticFiles from ../frontend (missing in the test environment) and
    initialises RAGSystem at module level (which connects to ChromaDB).
    The inline app is functionally identical for API testing purposes.
    """
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    test_app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class Source(BaseModel):
        label: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Source]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            answer, raw_sources = mock_rag_system.query(request.query, session_id)
            sources = [
                Source(**s) if isinstance(s, dict) else Source(label=s)
                for s in raw_sources
            ]
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        mock_rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return TestClient(test_app)
