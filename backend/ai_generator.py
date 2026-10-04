import anthropic
import httpx
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Max sequential rounds in which Claude may call tools for one user query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- Use the course outline tool for questions about a course's structure, outline, or lesson list; return the course title, course link, and every lesson's number and title
  - For each lesson, add a one-line summary written from its excerpt that names the specific techniques, tools, or examples covered
  - The summary must add information beyond the lesson title: never reuse the title's key words or paraphrase it (e.g. for "Database Indexing", write "B-tree vs. hash structures and when each speeds up lookups", not "Explains how to index a database")
- **Up to 2 sequential tool calls per query.** Make a second call only when it depends on the first call's result (e.g. get a course outline to find a lesson's title, then search for that topic in other courses). Prefer a single call whenever it is enough
  - Request independent lookups together in the same step, and never repeat a call with the same arguments
  - If a search yields no results, you may retry once with different terms; then state clearly that nothing was found, without offering alternatives
- Synthesize tool results into accurate, fact-based responses

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not narrate between tool calls
 - Do not mention "based on the search results"
- Format outlines exactly as:
    **Course Title:** <title>
    **Course Link:** <url>
    **Lessons:**
    - **Lesson <n>: <lesson title>**
        - <one-line summary>

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        # Short connect timeout so an unreachable server fails fast instead of hanging
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(60.0, connect=5.0),
        )
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Tools without a manager could never be executed - caller error
        if tools and tool_manager is None:
            raise ValueError("tool_manager is required when tools are provided")

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        # Get response from Claude
        response = self._create(messages, system_content, tools)

        # Keep going while Claude asks for tools (bounded by MAX_TOOL_ROUNDS)
        return self._run_tool_loop(
            response, messages, system_content, tools, tool_manager
        )

    def _create(self, messages: list, system: str, tools: Optional[List] = None):
        """Make one API request. Tools are offered only when given.

        Sends a copy of `messages` so later rounds never alter an earlier request.
        """
        params = {
            **self.base_params,
            "messages": list(messages),
            "system": system,
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = {"type": "auto"}
        return self.client.messages.create(**params)

    def _run_tool_loop(
        self, response, messages: list, system: str, tools: Optional[List], tool_manager
    ) -> str:
        """Execute tool requests round by round until Claude answers in text.

        Each round is a separate API request, so Claude sees earlier results before
        deciding on the next call. Stops when Claude's reply has no tool_use, after
        MAX_TOOL_ROUNDS rounds, or after a tool raises. In the last two cases one final
        request WITHOUT tools makes Claude answer from what it has.
        """
        rounds = 0
        while response.stop_reason == "tool_use" and tool_manager:
            tool_results, had_error = self._execute_tools(response, tool_manager)
            if not tool_results:
                break
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            rounds += 1

            if had_error or rounds >= self.MAX_TOOL_ROUNDS:
                return self._extract_text(self._create(messages, system))
            response = self._create(messages, system, tools)

        return self._extract_text(response)

    def _execute_tools(self, response, tool_manager) -> tuple[list[dict], bool]:
        """Run every tool_use block in the response.

        Returns (tool_result blocks in request order, whether any tool raised). A failure
        is reported to Claude as the tool result instead of failing the whole request.
        """
        tool_results = []
        had_error = False
        for content_block in response.content:
            if content_block.type == "tool_use":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                }
                try:
                    result_block["content"] = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )
                except Exception as e:
                    result_block["content"] = (
                        f"Error executing tool '{content_block.name}': {e}"
                    )
                    result_block["is_error"] = True
                    had_error = True
                tool_results.append(result_block)
        return tool_results, had_error

    @staticmethod
    def _extract_text(response) -> str:
        """Join the response's text blocks (it may also hold tool_use blocks)."""
        return "".join(b.text for b in response.content if b.type == "text")
