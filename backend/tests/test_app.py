"""HTTP layer (app.py): status codes and error mapping, with RAGSystem faked.

`app.py` builds a RAGSystem at import time and mounts ../frontend, so the `api`
fixture patches RAGSystem before importing app fresh, and runs from backend/.
TestClient is used WITHOUT a `with` block so the startup (doc ingestion) hook never runs.
"""
import sys
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import BACKEND_DIR

REQ = httpx.Request("POST", "http://llm.invalid")


@pytest.fixture
def api(monkeypatch):
    """Yields (TestClient, fake_rag) with a freshly imported app wired to the fake RAGSystem."""
    fake_rag = MagicMock()
    monkeypatch.setattr("rag_system.RAGSystem", lambda cfg: fake_rag)
    monkeypatch.chdir(BACKEND_DIR)
    sys.modules.pop("app", None)
    import app
    yield TestClient(app.app), fake_rag
    sys.modules.pop("app", None)


class TestQueryEndpoint:
    def test_success_creates_session_and_returns_answer_with_sources(self, api):
        client, rag = api
        rag.session_manager.create_session.return_value = "session_1"
        rag.query.return_value = ("ans", [{"title": "T", "link": None}])
        r = client.post("/api/query", json={"query": "hi"})
        assert r.status_code == 200
        assert r.json() == {"answer": "ans", "sources": [{"title": "T", "link": None}],
                            "session_id": "session_1"}
        rag.query.assert_called_once_with("hi", "session_1")

    def test_existing_session_is_reused(self, api):
        client, rag = api
        rag.query.return_value = ("ans", [])
        r = client.post("/api/query", json={"query": "hi", "session_id": "s9"})
        assert r.json()["session_id"] == "s9"
        rag.session_manager.create_session.assert_not_called()

    @pytest.mark.parametrize("error", [
        anthropic.APIConnectionError(request=REQ),
        anthropic.APITimeoutError(request=REQ),
    ])
    def test_unreachable_llm_gives_503_with_friendly_message(self, api, error):
        client, rag = api
        rag.query.side_effect = error
        r = client.post("/api/query", json={"query": "hi"})
        assert r.status_code == 503
        assert "unreachable" in r.json()["detail"]

    def test_other_errors_give_500_with_the_error_text(self, api):
        client, rag = api
        rag.query.side_effect = RuntimeError("boom")
        r = client.post("/api/query", json={"query": "hi"})
        assert r.status_code == 500 and r.json()["detail"] == "boom"


class TestCoursesEndpoint:
    def test_returns_course_stats(self, api):
        client, rag = api
        rag.get_course_analytics.return_value = {"total_courses": 2, "course_titles": ["A", "B"]}
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == {"total_courses": 2, "course_titles": ["A", "B"]}
