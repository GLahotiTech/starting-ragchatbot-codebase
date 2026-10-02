"""Does AIGenerator correctly offer / invoke / follow up on the search tool?

The Anthropic client is faked (see the `mock_anthropic` fixture in conftest.py), so
these tests check what AIGenerator SENDS and how it REACTS to canned replies.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ai_generator import AIGenerator
from search_tools import CourseSearchTool
from tests.conftest import text_response, tool_use_response

# The real tool definition (name/description/schema) that gets offered to Claude.
SEARCH_DEF = CourseSearchTool(MagicMock()).get_tool_definition()


def make_gen():
    """Build an AIGenerator. The mock_anthropic fixture has already swapped
    anthropic.Anthropic globally, so this gets the fake client automatically."""
    return AIGenerator("key", "model-x", base_url="http://x")


class TestToolOffering:
    """What gets sent to the API on the first request."""

    def test_tools_and_auto_choice_sent_only_when_provided(self, mock_anthropic):
        """With tools: request has tools, tool_choice=auto and the model. Without: neither key is sent."""
        mock_anthropic.messages.create.return_value = text_response()
        gen = make_gen()
        gen.generate_response("q", tools=[SEARCH_DEF], tool_manager=MagicMock())
        gen.generate_response("q")
        with_tools, without_tools = (c.kwargs for c in mock_anthropic.messages.create.call_args_list)
        assert with_tools["tools"] == [SEARCH_DEF]
        assert with_tools["tool_choice"] == {"type": "auto"}
        assert with_tools["model"] == "model-x"
        assert "tools" not in without_tools and "tool_choice" not in without_tools


class TestDirectAnswer:
    """Claude answers without using a tool."""

    def test_returns_text_without_tool_call(self, mock_anthropic):
        """stop_reason=end_turn -> return the text, run no tool, make exactly one API call."""
        mock_anthropic.messages.create.return_value = text_response("hello")
        tm = MagicMock()
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "hello"
        tm.execute_tool.assert_not_called()
        assert mock_anthropic.messages.create.call_count == 1


class TestToolExecution:
    """Claude asks for a tool; AIGenerator runs it and makes a follow-up call.

    `side_effect = [a, b]` makes the fake API return `a` on call 1 and `b` on call 2.
    """

    def test_executes_named_tool_with_model_arguments(self, mock_anthropic):
        """The tool Claude names is run with exactly the arguments Claude supplied."""
        mock_anthropic.messages.create.side_effect = [
            tool_use_response(tool_input={"query": "gizmo", "course_name": "Widgets", "lesson_number": 1}),
            text_response("done"),
        ]
        tm = MagicMock()
        tm.execute_tool.return_value = "TOOL OUTPUT"
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        tm.execute_tool.assert_called_once_with(
            "search_course_content", query="gizmo", course_name="Widgets", lesson_number=1)
        assert out == "done"

    def test_followup_call_contains_tool_result_and_no_tools(self, mock_anthropic):
        """2nd API call: no 'tools', and messages = [user query, assistant tool request, user tool_result]."""
        first = tool_use_response(tool_id="toolu_9")
        mock_anthropic.messages.create.side_effect = [first, text_response("done")]
        tm = MagicMock()
        tm.execute_tool.return_value = "TOOL OUTPUT"
        make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)

        assert mock_anthropic.messages.create.call_count == 2
        kw = mock_anthropic.messages.create.call_args_list[1].kwargs
        assert "tools" not in kw
        msgs = kw["messages"]
        assert msgs[1] == {"role": "assistant", "content": first.content}
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_9", "content": "TOOL OUTPUT"}]

    def test_multiple_tool_calls_in_one_reply_are_all_executed(self, mock_anthropic):
        """Claude may request several tools at once: run each, and return all results in ONE user message."""
        a = tool_use_response(name="search_course_content", tool_input={"query": "x"}, tool_id="id_a")
        b = tool_use_response(name="get_course_outline", tool_input={"course_title": "y"}, tool_id="id_b")
        both = SimpleNamespace(stop_reason="tool_use", content=a.content + b.content)
        mock_anthropic.messages.create.side_effect = [both, text_response("done")]
        tm = MagicMock()
        tm.execute_tool.side_effect = lambda name, **kw: f"result-{name}"

        assert make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm) == "done"

        assert [c.args[0] for c in tm.execute_tool.call_args_list] == [
            "search_course_content", "get_course_outline"]
        msgs = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"]
        assert len(msgs) == 3  # user query, assistant tool requests, ONE user message of results
        assert msgs[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "id_a", "content": "result-search_course_content"},
            {"type": "tool_result", "tool_use_id": "id_b", "content": "result-get_course_outline"},
        ]

    def test_tool_exception_is_reported_to_model_not_raised(self, mock_anthropic):
        """If a tool blows up, Claude should get the error as the tool result and still answer,
        rather than the whole request failing."""
        mock_anthropic.messages.create.side_effect = [tool_use_response(), text_response("sorry")]
        tm = MagicMock()
        tm.execute_tool.side_effect = RuntimeError("boom")
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "sorry"
        sent = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]
        assert "boom" in sent["content"]

    def test_tools_without_tool_manager_is_an_argument_error(self, mock_anthropic):
        """Passing tools but no tool_manager is a caller bug: raise before calling the API."""
        with pytest.raises(ValueError, match="tool_manager"):
            make_gen().generate_response("q", tools=[SEARCH_DEF])
        mock_anthropic.messages.create.assert_not_called()
