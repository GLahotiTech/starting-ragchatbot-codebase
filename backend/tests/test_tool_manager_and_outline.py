"""ToolManager (registry/dispatch) and CourseOutlineTool (course structure lookup)."""
from unittest.mock import MagicMock

import pytest

from config import Config
from models import Course, CourseChunk, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore


class TestToolManager:
    """Pure registry logic, using fake tools."""

    @staticmethod
    def fake_tool(name="fake", sources=None):
        tool = MagicMock()
        tool.get_tool_definition.return_value = {"name": name}
        tool.execute.return_value = f"{name}-result"
        tool.last_sources = sources or []
        return tool

    def test_register_and_list_definitions(self):
        tm = ToolManager()
        tm.register_tool(self.fake_tool("a"))
        tm.register_tool(self.fake_tool("b"))
        assert [d["name"] for d in tm.get_tool_definitions()] == ["a", "b"]

    def test_register_tool_without_name_is_rejected(self):
        tm = ToolManager()
        bad = MagicMock()
        bad.get_tool_definition.return_value = {}
        with pytest.raises(ValueError):
            tm.register_tool(bad)

    def test_execute_tool_dispatches_kwargs_to_the_named_tool(self):
        tm = ToolManager()
        tool = self.fake_tool("a")
        tm.register_tool(tool)
        assert tm.execute_tool("a", query="q", lesson_number=1) == "a-result"
        tool.execute.assert_called_once_with(query="q", lesson_number=1)

    def test_execute_unknown_tool_returns_message(self):
        assert ToolManager().execute_tool("nope") == "Tool 'nope' not found"

    def test_get_last_sources_then_reset(self):
        """Sources come from whichever tool has some; reset clears every tool."""
        tm = ToolManager()
        empty, filled = self.fake_tool("a"), self.fake_tool("b", sources=[{"title": "T", "link": None}])
        tm.register_tool(empty)
        tm.register_tool(filled)
        assert tm.get_last_sources() == [{"title": "T", "link": None}]
        tm.reset_sources()
        assert tm.get_last_sources() == [] and filled.last_sources == []


class TestCourseOutlineTool:
    """Uses the real_store fixture (Intro to Widgets + Gadget Basics)."""

    def test_outline_lists_title_link_and_all_lessons_in_order(self, real_store):
        out = CourseOutlineTool(real_store).execute("Intro to Widgets")
        lines = out.split("\n")
        assert lines[0] == "Course: Intro to Widgets"
        assert lines[1] == "Course Link: https://example.com/widgets"
        assert lines[2] == "Lessons (2):"
        lesson_lines = [l for l in lines if l.startswith("- Lesson")]
        assert lesson_lines == ["- Lesson 0: Overview", "- Lesson 1: Building"]

    def test_partial_name_resolves_to_the_right_course(self, real_store):
        tool = CourseOutlineTool(real_store)
        assert tool.execute("gadget").startswith("Course: Gadget Basics")
        assert tool.execute("widgets").startswith("Course: Intro to Widgets")

    def test_each_lesson_gets_an_excerpt_without_the_ingestion_prefix(self, real_store):
        out = CourseOutlineTool(real_store).execute("Intro to Widgets")
        assert "  Excerpt: Widgets are small reusable components." in out
        assert "  Excerpt: To build a widget, attach the gizmo to the sprocket." in out
        assert "content:" not in out  # the 'Lesson N content:' prefix is stripped

    def test_excerpt_samples_first_and_middle_chunk_of_long_lessons(self, tmp_path):
        store = VectorStore(str(tmp_path / "chroma"), Config.EMBEDDING_MODEL, 5)
        course = Course(title="Long", course_link="http://l", instructor="X",
                        lessons=[Lesson(lesson_number=0, title="Only", lesson_link=None)])
        store.add_course_metadata(course)
        store.add_course_content([
            CourseChunk(content=("Lesson 0 content: " if i == 0 else "") + f"chunk{i}",
                        course_title="Long", lesson_number=0, chunk_index=i)
            for i in range(5)])
        out = CourseOutlineTool(store).execute("Long")
        assert "  Excerpt: chunk0 ... chunk2" in out  # first, then index len//2 == 2

    def test_sets_last_sources_to_the_course(self, real_store):
        tool = CourseOutlineTool(real_store)
        tool.execute("Intro to Widgets")
        assert tool.last_sources == [{"title": "Intro to Widgets", "link": "https://example.com/widgets"}]

    def test_no_courses_in_catalog_gives_not_found_message(self, tmp_path):
        empty = VectorStore(str(tmp_path / "empty"), Config.EMBEDDING_MODEL, 5)
        assert CourseOutlineTool(empty).execute("anything") == "No course found matching 'anything'"
