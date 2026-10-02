"""RAGSystem handling of content-related queries.

Uses a real vector store (temp dir) and a FAKE Anthropic client; nothing here
touches the network or your real database.
"""
from unittest.mock import MagicMock

import pytest

from config import Config
from rag_system import RAGSystem
from tests.conftest import text_response, tool_use_response


@pytest.fixture
def rag(tmp_path, mock_anthropic, sample_course):
    """RAGSystem with real vector store (temp) and a faked Anthropic client.

    Fixture chaining: pytest first builds tmp_path, mock_anthropic and sample_course,
    then passes them here. A test that also asks for `mock_anthropic` receives the SAME
    fake client object that this RAGSystem uses, so it can script the replies.
    """
    cfg = Config(CHROMA_PATH=str(tmp_path / "chroma"), ANTHROPIC_API_KEY="k")
    system = RAGSystem(cfg)
    course, chunks = sample_course
    system.vector_store.add_course_metadata(course)
    system.vector_store.add_course_content(chunks)
    return system


class TestContentQuery:
    """RAGSystem.query() end to end, with scripted Claude replies."""

    def test_content_query_returns_answer_and_sources(self, rag, mock_anthropic):
        """Full flow: Claude calls search -> real DB lookup -> final answer + sources with links."""
        mock_anthropic.messages.create.side_effect = [
            tool_use_response(tool_input={"query": "build a widget", "lesson_number": 1}),
            text_response("Attach the gizmo."),
        ]
        answer, sources = rag.query("How do I build a widget?")
        assert answer == "Attach the gizmo."
        assert sources == [{"title": "Intro to Widgets - Lesson 1",
                            "link": "https://example.com/widgets/1"}]
        # The real search result (not just the final answer) reached the follow-up call
        tool_result = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]["content"]
        assert "gizmo" in tool_result

    def test_sources_reset_after_query(self, rag, mock_anthropic):
        """Sources from query 1 must not leak into query 2 (which does no search)."""
        mock_anthropic.messages.create.side_effect = [
            tool_use_response(), text_response("a"), text_response("general answer")]
        rag.query("content q")
        _, sources = rag.query("general q")
        assert sources == []

    def test_prompt_wrapped_and_both_tools_offered(self, rag, mock_anthropic):
        """The user's text is in the prompt sent to Claude, and both tools are registered and offered."""
        mock_anthropic.messages.create.return_value = text_response("x")
        rag.query("What is a widget?")
        kw = mock_anthropic.messages.create.call_args.kwargs
        assert "What is a widget?" in kw["messages"][0]["content"]
        assert [t["name"] for t in kw["tools"]] == ["search_course_content", "get_course_outline"]

    def test_session_history_is_recorded_and_reused(self, rag, mock_anthropic):
        """The 2nd query in a session sees the 1st question and answer in its system prompt."""
        mock_anthropic.messages.create.return_value = text_response("first answer")
        sid = rag.session_manager.create_session()
        rag.query("first question", sid)
        rag.query("second question", sid)
        system = mock_anthropic.messages.create.call_args.kwargs["system"]
        assert "first question" in system and "first answer" in system

    def test_outline_query_uses_outline_tool(self, rag, mock_anthropic):
        """When Claude picks get_course_outline, the lesson list is returned and a source recorded."""
        mock_anthropic.messages.create.side_effect = [
            tool_use_response(name="get_course_outline", tool_input={"course_title": "Widgets"}),
            text_response("outline"),
        ]
        answer, sources = rag.query("Outline of the widgets course")
        tool_result = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]["content"]
        assert "Lesson 1: Building" in tool_result
        assert sources[0]["title"] == "Intro to Widgets"

    def test_search_tool_failure_is_surfaced_to_model_not_raised(self, rag, mock_anthropic):
        """A broken vector store should yield a tool-result string, so the model can answer."""
        rag.vector_store.course_content = MagicMock()
        rag.vector_store.course_content.query.side_effect = RuntimeError("chroma down")
        mock_anthropic.messages.create.side_effect = [tool_use_response(), text_response("sorry")]
        answer, _ = rag.query("content q")
        assert answer == "sorry"
        result = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]["content"]
        assert "chroma down" in result


class TestIngestion:
    """Loading documents from a folder."""

    GADGET_DOC = ("Course Title: Gadget Basics\nCourse Link: http://g\nCourse Instructor: Bo\n\n"
                  "Lesson 0: Intro\nLesson Link: http://g/0\nGadgets whir and click when powered.\n")

    def test_add_course_folder_then_search(self, rag, tmp_path):
        """Write a .txt course in the expected format, ingest it, and find its content via search."""
        d = tmp_path / "docs"
        d.mkdir()
        (d / "c.txt").write_text(self.GADGET_DOC)
        courses, chunks = rag.add_course_folder(str(d))
        assert courses == 1 and chunks >= 1
        out = rag.search_tool.execute("gadgets whir", course_name="Gadget")
        assert "whir" in out

    def test_second_ingest_skips_existing_courses(self, rag, tmp_path):
        """Re-running ingestion (as happens on every server start) must not duplicate a course."""
        d = tmp_path / "docs"
        d.mkdir()
        (d / "c.txt").write_text(self.GADGET_DOC)
        rag.add_course_folder(str(d))
        titles_before = sorted(rag.vector_store.get_existing_course_titles())
        chunks_before = rag.vector_store.course_content.count()

        assert rag.add_course_folder(str(d)) == (0, 0)
        assert sorted(rag.vector_store.get_existing_course_titles()) == titles_before
        assert rag.vector_store.course_content.count() == chunks_before
