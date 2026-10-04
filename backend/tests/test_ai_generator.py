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
        with_tools, without_tools = (
            c.kwargs for c in mock_anthropic.messages.create.call_args_list
        )
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
            tool_use_response(
                tool_input={
                    "query": "gizmo",
                    "course_name": "Widgets",
                    "lesson_number": 1,
                }
            ),
            text_response("done"),
        ]
        tm = MagicMock()
        tm.execute_tool.return_value = "TOOL OUTPUT"
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        tm.execute_tool.assert_called_once_with(
            "search_course_content",
            query="gizmo",
            course_name="Widgets",
            lesson_number=1,
        )
        assert out == "done"

    def test_followup_call_contains_tool_result_and_still_offers_tools(
        self, mock_anthropic
    ):
        """2nd API call: messages = [user query, assistant tool request, user tool_result], and tools
        are still offered so Claude may chain a second tool call."""
        first = tool_use_response(tool_id="toolu_9")
        mock_anthropic.messages.create.side_effect = [first, text_response("done")]
        tm = MagicMock()
        tm.execute_tool.return_value = "TOOL OUTPUT"
        make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)

        assert mock_anthropic.messages.create.call_count == 2
        kw = mock_anthropic.messages.create.call_args_list[1].kwargs
        assert kw["tools"] == [SEARCH_DEF]
        msgs = kw["messages"]
        assert msgs[1] == {"role": "assistant", "content": first.content}
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_9", "content": "TOOL OUTPUT"}
        ]

    def test_multiple_tool_calls_in_one_reply_are_all_executed(self, mock_anthropic):
        """Claude may request several tools at once: run each, and return all results in ONE user message."""
        a = tool_use_response(
            name="search_course_content", tool_input={"query": "x"}, tool_id="id_a"
        )
        b = tool_use_response(
            name="get_course_outline", tool_input={"course_title": "y"}, tool_id="id_b"
        )
        both = SimpleNamespace(stop_reason="tool_use", content=a.content + b.content)
        mock_anthropic.messages.create.side_effect = [both, text_response("done")]
        tm = MagicMock()
        tm.execute_tool.side_effect = lambda name, **kw: f"result-{name}"

        assert (
            make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
            == "done"
        )

        assert [c.args[0] for c in tm.execute_tool.call_args_list] == [
            "search_course_content",
            "get_course_outline",
        ]
        msgs = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"]
        assert (
            len(msgs) == 3
        )  # user query, assistant tool requests, ONE user message of results
        assert msgs[2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "id_a",
                "content": "result-search_course_content",
            },
            {
                "type": "tool_result",
                "tool_use_id": "id_b",
                "content": "result-get_course_outline",
            },
        ]

    def test_tool_exception_is_reported_to_model_not_raised(self, mock_anthropic):
        """If a tool blows up, Claude should get the error as the tool result and still answer,
        rather than the whole request failing."""
        mock_anthropic.messages.create.side_effect = [
            tool_use_response(),
            text_response("sorry"),
        ]
        tm = MagicMock()
        tm.execute_tool.side_effect = RuntimeError("boom")
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "sorry"
        sent = mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][2][
            "content"
        ][0]
        assert "boom" in sent["content"]

    def test_tools_without_tool_manager_is_an_argument_error(self, mock_anthropic):
        """Passing tools but no tool_manager is a caller bug: raise before calling the API."""
        with pytest.raises(ValueError, match="tool_manager"):
            make_gen().generate_response("q", tools=[SEARCH_DEF])
        mock_anthropic.messages.create.assert_not_called()


def outline_use(tool_id="toolu_o"):
    return tool_use_response(
        name="get_course_outline", tool_input={"course_title": "X"}, tool_id=tool_id
    )


def search_use(tool_id="toolu_s"):
    return tool_use_response(
        name="search_course_content", tool_input={"query": "topic"}, tool_id=tool_id
    )


def tools_offered(mock):
    """For each API call made, whether it was offered tools."""
    return ["tools" in c.kwargs for c in mock.messages.create.call_args_list]


class TestSequentialToolCalls:
    """Claude may chain up to 2 tool rounds, each as its own API request."""

    def test_two_rounds_chain_results_into_final_answer(self, mock_anthropic):
        """outline -> search -> answer: 3 API calls, tools run in order, each round sees prior results."""
        mock_anthropic.messages.create.side_effect = [
            outline_use(),
            search_use(),
            text_response("Course Y covers it."),
        ]
        tm = MagicMock()
        tm.execute_tool.side_effect = lambda name, **kw: f"result-{name}"

        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)

        assert out == "Course Y covers it."
        assert [(c.args[0], c.kwargs) for c in tm.execute_tool.call_args_list] == [
            ("get_course_outline", {"course_title": "X"}),
            ("search_course_content", {"query": "topic"}),
        ]
        calls = mock_anthropic.messages.create.call_args_list
        assert len(calls) == 3
        # Round 2 request already contains round 1's result
        assert (
            calls[1].kwargs["messages"][2]["content"][0]["content"]
            == "result-get_course_outline"
        )
        # Final request carries the whole conversation: query, 2 x (tool request, tool result)
        final = calls[2].kwargs["messages"]
        assert [m["role"] for m in final] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert final[4]["content"][0]["content"] == "result-search_course_content"
        assert final[4]["content"][0]["tool_use_id"] == "toolu_s"

    def test_tools_offered_on_both_rounds_but_not_on_final_call(self, mock_anthropic):
        """Calls 1 and 2 offer tools; the synthesis call after round 2 does not."""
        mock_anthropic.messages.create.side_effect = [
            outline_use(),
            search_use(),
            text_response("ok"),
        ]
        make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=MagicMock())
        assert tools_offered(mock_anthropic) == [True, True, False]

    def test_stops_when_reply_has_no_tool_use(self, mock_anthropic):
        """If Claude answers after round 1, no third request is made."""
        mock_anthropic.messages.create.side_effect = [
            search_use(),
            text_response("done"),
        ]
        tm = MagicMock()
        make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert mock_anthropic.messages.create.call_count == 2
        assert tm.execute_tool.call_count == 1

    def test_never_runs_more_than_two_rounds(self, mock_anthropic):
        """Even if Claude keeps asking for tools, only 2 rounds run and the final call has no tools."""
        mock_anthropic.messages.create.side_effect = [
            search_use("a"),
            search_use("b"),
            text_response("answer anyway"),
            search_use("c"),
        ]
        tm = MagicMock()
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "answer anyway"
        assert tm.execute_tool.call_count == 2
        assert mock_anthropic.messages.create.call_count == 3
        assert tools_offered(mock_anthropic) == [True, True, False]

    def test_earlier_requests_are_not_altered_by_later_rounds(self, mock_anthropic):
        """The first request still shows only the user query after the whole flow has finished."""
        mock_anthropic.messages.create.side_effect = [
            outline_use(),
            search_use(),
            text_response("ok"),
        ]
        make_gen().generate_response(
            "the question", tools=[SEARCH_DEF], tool_manager=MagicMock()
        )
        calls = mock_anthropic.messages.create.call_args_list
        assert [len(c.kwargs["messages"]) for c in calls] == [1, 3, 5]
        assert calls[0].kwargs["messages"][0]["content"] == "the question"

    def test_system_prompt_and_history_sent_on_every_call(self, mock_anthropic):
        """Conversation history in the system prompt is preserved across rounds."""
        mock_anthropic.messages.create.side_effect = [
            outline_use(),
            search_use(),
            text_response("ok"),
        ]
        make_gen().generate_response(
            "q",
            conversation_history="User: hi\nAssistant: hello",
            tools=[SEARCH_DEF],
            tool_manager=MagicMock(),
        )
        systems = [
            c.kwargs["system"] for c in mock_anthropic.messages.create.call_args_list
        ]
        assert len(set(systems)) == 1
        assert "User: hi" in systems[0]

    def test_parallel_tool_uses_count_as_one_round(self, mock_anthropic):
        """Two tool_use blocks in the first reply are one round, so a second round is still allowed."""
        both = SimpleNamespace(
            stop_reason="tool_use",
            content=outline_use("id_a").content + search_use("id_b").content,
        )
        mock_anthropic.messages.create.side_effect = [
            both,
            search_use("id_c"),
            text_response("ok"),
        ]
        tm = MagicMock()
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "ok"
        assert tm.execute_tool.call_count == 3
        assert tools_offered(mock_anthropic) == [True, True, False]

    def test_text_alongside_tool_use_is_kept_in_history_and_not_returned_early(
        self, mock_anthropic
    ):
        """A reply with a text preamble plus a tool_use still triggers the tool; the final text is returned."""
        mixed = SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="Let me look."),
                *search_use().content,
            ],
        )
        mock_anthropic.messages.create.side_effect = [mixed, text_response("final")]
        tm = MagicMock()
        assert (
            make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
            == "final"
        )
        tm.execute_tool.assert_called_once()
        assert (
            mock_anthropic.messages.create.call_args_list[1].kwargs["messages"][1][
                "content"
            ]
            == mixed.content
        )


class TestToolFailureInSequence:
    """A raising tool ends the chain; Claude is told and answers without further tools."""

    @pytest.mark.parametrize("failing_call", [1, 2])
    def test_exception_stops_further_tool_rounds(self, mock_anthropic, failing_call):
        """Whichever round fails, no more tools run and the final request offers no tools."""
        replies = (
            [outline_use(), text_response("sorry")]
            if failing_call == 1
            else [outline_use(), search_use(), text_response("sorry")]
        )
        mock_anthropic.messages.create.side_effect = replies + [search_use("never")]
        tm = MagicMock()
        results = iter(
            [RuntimeError("boom"), "fine"]
            if failing_call == 1
            else ["fine", RuntimeError("boom")]
        )

        def execute(name, **kw):
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return r

        tm.execute_tool.side_effect = execute

        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)

        assert out == "sorry"
        assert tm.execute_tool.call_count == failing_call
        assert mock_anthropic.messages.create.call_count == failing_call + 1
        last = mock_anthropic.messages.create.call_args.kwargs
        assert "tools" not in last
        error_block = last["messages"][-1]["content"][0]
        assert error_block["is_error"] is True and "boom" in error_block["content"]

    def test_round_one_results_survive_a_round_two_failure(self, mock_anthropic):
        """When round 2 fails, the final request still contains round 1's successful result."""
        mock_anthropic.messages.create.side_effect = [
            outline_use(),
            search_use(),
            text_response("partial"),
        ]
        tm = MagicMock()
        tm.execute_tool.side_effect = ["OUTLINE OK", RuntimeError("boom")]
        make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        msgs = mock_anthropic.messages.create.call_args.kwargs["messages"]
        assert msgs[2]["content"][0]["content"] == "OUTLINE OK"

    def test_error_string_result_is_not_a_failure(self, mock_anthropic):
        """A tool that RETURNS an error/no-results string is a normal result: Claude can retry in round 2."""
        mock_anthropic.messages.create.side_effect = [
            search_use("a"),
            search_use("b"),
            text_response("found"),
        ]
        tm = MagicMock()
        tm.execute_tool.side_effect = ["No relevant content found.", "Real content"]
        out = make_gen().generate_response("q", tools=[SEARCH_DEF], tool_manager=tm)
        assert out == "found"
        assert tm.execute_tool.call_count == 2


class TestSystemPrompt:
    def test_allows_two_sequential_tool_calls(self):
        """The prompt no longer forbids a second tool call."""
        assert "One tool call per query" not in AIGenerator.SYSTEM_PROMPT
        assert "2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT
