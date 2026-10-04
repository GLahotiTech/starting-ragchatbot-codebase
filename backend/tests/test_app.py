"""HTTP layer (app.py): status codes and error mapping, with RAGSystem faked.

The `api` fixture (conftest.py) imports app.py with RAGSystem faked and returns a TestClient.
"""
import anthropic
import httpx
import pytest

REQ = httpx.Request("POST", "http://llm.invalid")


class TestQueryEndpoint:
    def test_success_creates_session_and_returns_answer_with_sources(self, api):
        client, rag = api
        rag.session_manager.create_session.return_value = "session_1"
        rag.query.return_value = ("ans", [{"title": "T", "link": None}])
        r = client.post("/api/query", json={"query": "hi"})
        assert r.status_code == 200
        assert r.json() == {
            "answer": "ans",
            "sources": [{"title": "T", "link": None}],
            "session_id": "session_1",
        }
        rag.query.assert_called_once_with("hi", "session_1")

    def test_existing_session_is_reused(self, api):
        client, rag = api
        rag.query.return_value = ("ans", [])
        r = client.post("/api/query", json={"query": "hi", "session_id": "s9"})
        assert r.json()["session_id"] == "s9"
        rag.session_manager.create_session.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [
            anthropic.APIConnectionError(request=REQ),
            anthropic.APITimeoutError(request=REQ),
        ],
    )
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
        rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["A", "B"],
        }
        r = client.get("/api/courses")
        assert r.status_code == 200
        assert r.json() == {"total_courses": 2, "course_titles": ["A", "B"]}

    def test_empty_analytics(self, api):
        client, rag = api
        rag.get_course_analytics.return_value = {"total_courses": 0, "course_titles": []}
        assert client.get("/api/courses").json() == {"total_courses": 0, "course_titles": []}

    def test_failure_gives_500(self, api):
        client, rag = api
        rag.get_course_analytics.side_effect = RuntimeError("db down")
        r = client.get("/api/courses")
        assert r.status_code == 500 and r.json()["detail"] == "db down"


class TestRequestValidation:
    @pytest.mark.parametrize("body", [{}, {"session_id": "s"}, {"query": 123}, {"query": None}])
    def test_bad_query_body_gives_422(self, api, body):
        client, rag = api
        assert client.post("/api/query", json=body).status_code == 422
        rag.query.assert_not_called()

    def test_query_rejects_get(self, api):
        client, _ = api
        # The "/" static mount catches the unmatched GET, so it is a 404 rather than 405.
        assert client.get("/api/query").status_code in (404, 405)

    def test_courses_rejects_post(self, api):
        client, _ = api
        assert client.post("/api/courses").status_code == 405

    def test_sources_with_link_round_trip(self, api):
        client, rag = api
        rag.query.return_value = ("a", [{"title": "T", "link": "https://x.test"}])
        r = client.post("/api/query", json={"query": "q", "session_id": "s"})
        assert r.json()["sources"] == [{"title": "T", "link": "https://x.test"}]


class TestRootAndStatic:
    def test_root_serves_frontend_html(self, api):
        client, _ = api
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_unknown_path_is_404(self, api):
        client, _ = api
        assert client.get("/nope.xyz").status_code == 404

    def test_cors_headers_present(self, api):
        client, _ = api
        r = client.get("/api/courses", headers={"Origin": "http://example.com"})
        assert r.headers.get("access-control-allow-origin") in ("*", "http://example.com")
