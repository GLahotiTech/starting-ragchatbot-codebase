"""Tests for CourseSearchTool.execute, in two layers:
  1. TestExecuteUnit            - mocked VectorStore: tests execute()'s own logic only
  2. TestExecuteIntegration     - real ChromaDB + embeddings on tiny sample data (2 courses)
(Checks against the real database live in backend/live_checks.py, not here.)
"""
from unittest.mock import MagicMock

import pytest

from search_tools import CourseSearchTool
from vector_store import SearchResults


def _results(docs, metas):
    """Helper: build a SearchResults with dummy distances."""
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs))


# ---------- unit: mocked VectorStore ----------
class TestExecuteUnit:
    """The store is a MagicMock returning canned results; no database involved."""

    def make(self, results):
        """Build a tool around a fake store whose search() returns `results`.

        Returns (tool, store) so tests can also inspect calls made on the store.
        """
        store = MagicMock()
        store.search.return_value = results
        store.get_lesson_link.return_value = "http://lesson"
        store.get_course_link.return_value = "http://course"
        return CourseSearchTool(store), store

    def test_formats_results_with_header(self):
        """Output is '[Course - Lesson N]' on one line, then the chunk text."""
        tool, _ = self.make(_results(["body"], [{"course_title": "C", "lesson_number": 1}]))
        assert tool.execute("q") == "[C - Lesson 1]\nbody"

    def test_error_is_returned_verbatim(self):
        """If the store reports an error, execute() returns that error string as-is."""
        tool, _ = self.make(SearchResults.empty("Search error: boom"))
        assert tool.execute("q") == "Search error: boom"

    @pytest.mark.parametrize("course, lesson, expected", [
        (None, None, "No relevant content found."),
        ("X", None, "No relevant content found in course 'X'."),
        (None, 3, "No relevant content found in lesson 3."),
        ("X", 3, "No relevant content found in course 'X' in lesson 3."),
    ])
    def test_empty_results_message_names_the_filters(self, course, lesson, expected):
        """No hits -> message names whichever filters were applied."""
        tool, _ = self.make(_results([], []))
        assert tool.execute("q", course_name=course, lesson_number=lesson) == expected

    def test_empty_results_message_mentions_lesson_zero(self):
        """Lesson 0 is a valid lesson but falsy: the empty-result message must still name it."""
        tool, _ = self.make(_results([], []))
        assert tool.execute("q", lesson_number=0) == "No relevant content found in lesson 0."

    def test_sources_tracked_and_deduplicated(self):
        """last_sources has one entry per distinct (course, lesson), even with 2 chunks from one lesson."""
        metas = [{"course_title": "C", "lesson_number": 1}] * 2 + [{"course_title": "C", "lesson_number": 2}]
        tool, _ = self.make(_results(["a", "b", "c"], metas))
        tool.execute("q")
        assert [s["title"] for s in tool.last_sources] == ["C - Lesson 1", "C - Lesson 2"]
        assert tool.last_sources[0]["link"] == "http://lesson"

    def test_source_link_falls_back_to_course_link(self):
        """If a lesson has no link, the source uses the course-level link instead."""
        tool, store = self.make(_results(["a"], [{"course_title": "C", "lesson_number": 1}]))
        store.get_lesson_link.return_value = None
        tool.execute("q")
        assert tool.last_sources[0]["link"] == "http://course"

    def test_lesson_zero_is_not_dropped(self):
        """Guards against `if lesson_number:` bugs in the formatting path: lesson 0 is valid but falsy."""
        tool, _ = self.make(_results(["a"], [{"course_title": "C", "lesson_number": 0}]))
        assert "[C - Lesson 0]" in tool.execute("q")


# ---------- integration: real ChromaDB + embeddings (temp dir) ----------
class TestExecuteIntegration:
    """Uses the `real_store` fixture: two courses (Intro to Widgets, Gadget Basics), 2 chunks each."""

    def test_content_query_ranks_relevant_chunk_first(self, real_store):
        """A natural-language question puts the right chunk first, ahead of other courses' chunks."""
        out = CourseSearchTool(real_store).execute("how do I build a widget")
        assert out.startswith("[Intro to Widgets - Lesson 1]")
        assert "gizmo" in out.split("\n\n")[0]

    def test_course_name_filter_resolves_partial_name_to_the_right_course(self, real_store):
        """A partial name picks the matching course and excludes the other course's content,
        even when the query text better matches the other course."""
        tool = CourseSearchTool(real_store)
        out = tool.execute("battery", course_name="widgets")
        assert "Intro to Widgets" in out and "Gadget Basics" not in out
        out = tool.execute("components", course_name="gadgets")
        assert "Gadget Basics" in out and "Intro to Widgets" not in out

    def test_lesson_filter_only_returns_that_lesson(self, real_store):
        """lesson_number=0 (no course given) returns lesson 0 content only."""
        out = CourseSearchTool(real_store).execute("widget", lesson_number=0)
        assert "Lesson 0" in out and "Lesson 1" not in out

    def test_course_and_lesson_filter_combined(self, real_store):
        """Uses the $and filter path: only that course's lesson comes back."""
        out = CourseSearchTool(real_store).execute("widget", course_name="Gadget", lesson_number=1)
        assert "flux capacitor" in out
        assert "Intro to Widgets" not in out and "Lesson 0" not in out

    def test_populates_sources_with_links(self, real_store):
        """After a filtered search, last_sources holds exactly that lesson with its real URL."""
        tool = CourseSearchTool(real_store)
        tool.execute("build a widget", course_name="Widgets", lesson_number=1)
        assert tool.last_sources == [{"title": "Intro to Widgets - Lesson 1",
                                      "link": "https://example.com/widgets/1"}]
