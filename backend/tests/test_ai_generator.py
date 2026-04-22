"""
test_ai_generator.py
====================
Unit tests for AIGenerator in backend/ai_generator.py.

WHAT IS BEING TESTED
--------------------
AIGenerator wraps the Anthropic API. Its job is to:
  1. Build the correct API request (system prompt, messages, tools).
  2. Detect when Claude wants to call a tool (stop_reason == "tool_use").
  3. Execute up to 2 sequential tool-calling rounds via the ToolManager.
  4. Make a final synthesis call (without tools) after tool rounds complete.
  5. Return Claude's text response.

HOW THESE TESTS WORK
---------------------
Every test mocks `self.client.messages.create` — no real API calls are made.
We feed the mock a `side_effect` list: the first item is returned on the first
call, the second on the second call, and so on. This lets us simulate any
combination of tool-use and text responses from the API.

`tool_manager` is also a MagicMock so we can inspect which tools were called
and control what they return.

KEY TERMS
---------
- stop_reason="end_turn"  → Claude finished and wrote a text answer.
- stop_reason="tool_use"  → Claude wants to call a tool before answering.
- Intermediate call        → API call made between rounds, with tools still included.
- Synthesis call           → Final API call, no tools, Claude writes the answer.

PYTEST QUICK REFERENCE
-----------------------
- test_ prefix → auto-discovered and run.
- @pytest.fixture → reusable setup, injected as function arguments.
- MagicMock       → fake object; all method calls succeed and return more mocks.
- side_effect=[]  → each call to the mock returns the next item in the list.
- assert_called_once_with(args) → verifies a mock was called exactly once.
- call_args_list[n][1] → keyword args of the nth call to a mock.
- Run: uv run pytest backend/tests/test_ai_generator.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Mock-building helpers
# ---------------------------------------------------------------------------

def text_block(text):
    """Simulate an Anthropic TextBlock (has a .text attribute)."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def tool_use_block(name, input_dict, tool_id="tid_1"):
    """Simulate an Anthropic ToolUseBlock (has .type, .name, .input, .id)."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    block.id = tool_id
    return block


def api_response(stop_reason, *blocks):
    """
    Simulate a response from client.messages.create().

    Args:
        stop_reason – "end_turn" or "tool_use"
        *blocks     – content blocks returned in response.content
    """
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = list(blocks)
    return resp


FAKE_TOOLS = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    """AIGenerator with a fake API key — no real HTTP calls are made."""
    return AIGenerator(api_key="sk-fake", model="claude-test")


@pytest.fixture
def tool_manager():
    """A fake ToolManager whose execute_tool returns a fixed string by default."""
    mgr = MagicMock()
    mgr.execute_tool.return_value = "tool result content"
    return mgr


# ---------------------------------------------------------------------------
# Group A — Direct text response (no tools invoked)
# ---------------------------------------------------------------------------

class TestDirectResponse:
    """
    When Claude answers without calling any tool, generate_response should
    return the text immediately with a single API call.
    """

    def test_returns_text_on_end_turn(self, generator):
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("Hello!"))):
            assert generator.generate_response("hi") == "Hello!"

    def test_single_api_call_when_no_tool_use(self, generator):
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("ok"))) as mock_create:
            generator.generate_response("test")
            assert mock_create.call_count == 1

    def test_execute_tool_never_called_on_direct_response(self, generator, tool_manager):
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("ok"))):
            generator.generate_response("test", tools=FAKE_TOOLS, tool_manager=tool_manager)
            tool_manager.execute_tool.assert_not_called()

    def test_tools_added_to_api_call_when_provided(self, generator):
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("ok"))) as mock_create:
            generator.generate_response("test", tools=FAKE_TOOLS)
            kwargs = mock_create.call_args[1]
            assert kwargs["tools"] == FAKE_TOOLS
            assert kwargs["tool_choice"] == {"type": "auto"}

    def test_conversation_history_injected_into_system_prompt(self, generator):
        history = "User: hi\nAssistant: hello"
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("ok"))) as mock_create:
            generator.generate_response("follow-up", conversation_history=history)
            system = mock_create.call_args[1]["system"]
            assert "Previous conversation:" in system
            assert history in system

    def test_no_tools_key_in_api_call_when_tools_not_provided(self, generator):
        with patch.object(generator.client.messages, "create",
                          return_value=api_response("end_turn", text_block("ok"))) as mock_create:
            generator.generate_response("test")
            kwargs = mock_create.call_args[1]
            assert "tools" not in kwargs
            assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# Group B — Single tool round, Claude answers on intermediate call
# ---------------------------------------------------------------------------

class TestSingleToolRound:
    """
    Claude calls one tool (round 1), then returns text on the intermediate
    API call. No synthesis call should be made — the intermediate text is
    returned directly.

    API call sequence: initial (tool_use) → intermediate (end_turn)
    Total calls: 2
    """

    def test_execute_tool_called_once_with_correct_args(self, generator, tool_manager):
        """The tool name and input from the tool_use block must be forwarded exactly."""
        initial = api_response("tool_use",
                               tool_use_block("search_course_content", {"query": "what is RAG"}))
        intermediate = api_response("end_turn", text_block("RAG is..."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate]):
            generator.generate_response("what is RAG?", tools=FAKE_TOOLS,
                                        tool_manager=tool_manager)

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="what is RAG"
        )

    def test_returns_intermediate_text_directly(self, generator, tool_manager):
        """When the intermediate call returns text, that text is the final result."""
        initial = api_response("tool_use", tool_use_block("search_course_content", {"query": "q"}))
        intermediate = api_response("end_turn", text_block("The answer is 42."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate]):
            result = generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert result == "The answer is 42."

    def test_exactly_two_api_calls_for_single_round(self, generator, tool_manager):
        """Single round = initial call + one intermediate call (no synthesis call)."""
        initial = api_response("tool_use", tool_use_block("search_course_content", {"query": "q"}))
        intermediate = api_response("end_turn", text_block("answer"))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert mock_create.call_count == 2

    def test_intermediate_call_includes_tools(self, generator, tool_manager):
        """
        The intermediate API call must still include tools so Claude can chain
        a second tool call if needed.
        """
        initial = api_response("tool_use", tool_use_block("search_course_content", {"query": "q"}))
        intermediate = api_response("end_turn", text_block("answer"))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        intermediate_kwargs = mock_create.call_args_list[1][1]
        assert "tools" in intermediate_kwargs


# ---------------------------------------------------------------------------
# Group C — Two tool rounds, synthesis call needed
# ---------------------------------------------------------------------------

class TestTwoToolRounds:
    """
    Claude calls a tool in round 1, sees the result, then calls a second tool
    in round 2. After both rounds, a final synthesis call (no tools) is made.

    API call sequence: initial (tool_use) → intermediate (tool_use) → synthesis (end_turn)
    Total calls: 3
    """

    def test_execute_tool_called_twice(self, generator, tool_manager):
        """Both tool rounds must invoke execute_tool."""
        initial = api_response("tool_use",
                               tool_use_block("get_course_outline", {"course_name": "MCP"}, "t1"))
        intermediate = api_response("tool_use",
                                    tool_use_block("search_course_content",
                                                   {"query": "lesson 4 topic"}, "t2"))
        synthesis = api_response("end_turn", text_block("Here is the answer."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate, synthesis]):
            generator.generate_response("complex query", tools=FAKE_TOOLS,
                                        tool_manager=tool_manager)

        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="MCP")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="lesson 4 topic")

    def test_three_api_calls_total(self, generator, tool_manager):
        """Two tool rounds plus one synthesis = 3 API calls."""
        initial = api_response("tool_use", tool_use_block("get_course_outline",
                                                          {"course_name": "MCP"}, "t1"))
        intermediate = api_response("tool_use", tool_use_block("search_course_content",
                                                                {"query": "topic"}, "t2"))
        synthesis = api_response("end_turn", text_block("Final answer."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate, synthesis]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert mock_create.call_count == 3

    def test_synthesis_call_has_no_tools(self, generator, tool_manager):
        """
        The synthesis call (3rd API call) must NOT include 'tools' or 'tool_choice'.
        This prevents Claude from calling more tools in the synthesis step.
        """
        initial = api_response("tool_use",
                               tool_use_block("get_course_outline", {"course_name": "X"}, "t1"))
        intermediate = api_response("tool_use",
                                    tool_use_block("search_course_content", {"query": "q"}, "t2"))
        synthesis = api_response("end_turn", text_block("Done."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate, synthesis]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        synthesis_kwargs = mock_create.call_args_list[2][1]
        assert "tools" not in synthesis_kwargs
        assert "tool_choice" not in synthesis_kwargs

    def test_returns_synthesis_text(self, generator, tool_manager):
        """After two tool rounds the returned string comes from the synthesis call."""
        initial = api_response("tool_use",
                               tool_use_block("get_course_outline", {"course_name": "X"}, "t1"))
        intermediate = api_response("tool_use",
                                    tool_use_block("search_course_content", {"query": "q"}, "t2"))
        synthesis = api_response("end_turn", text_block("Synthesis answer!"))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, intermediate, synthesis]):
            result = generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert result == "Synthesis answer!"


# ---------------------------------------------------------------------------
# Group D — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """
    When a tool raises an exception, the loop should stop and still proceed to
    a synthesis call so Claude can write a graceful response rather than crashing.
    """

    def test_tool_exception_does_not_propagate(self, generator, tool_manager):
        """A RuntimeError inside execute_tool must NOT bubble up to the caller."""
        tool_manager.execute_tool.side_effect = RuntimeError("DB offline")
        initial = api_response("tool_use",
                               tool_use_block("search_course_content", {"query": "q"}))
        synthesis = api_response("end_turn", text_block("Sorry, something went wrong."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, synthesis]):
            result = generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert isinstance(result, str)

    def test_tool_exception_stops_loop_and_proceeds_to_synthesis(self, generator, tool_manager):
        """
        After a tool error in round 1, there must be no second intermediate
        call (round 2 is skipped). Only the synthesis call follows.
        Total API calls: 2 (initial + synthesis).
        """
        tool_manager.execute_tool.side_effect = RuntimeError("DB offline")
        initial = api_response("tool_use",
                               tool_use_block("search_course_content", {"query": "q"}))
        synthesis = api_response("end_turn", text_block("Graceful response."))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, synthesis]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        assert mock_create.call_count == 2

    def test_synthesis_after_error_has_no_tools(self, generator, tool_manager):
        """Even the error-path synthesis call must omit tools."""
        tool_manager.execute_tool.side_effect = RuntimeError("fail")
        initial = api_response("tool_use",
                               tool_use_block("search_course_content", {"query": "q"}))
        synthesis = api_response("end_turn", text_block("ok"))

        with patch.object(generator.client.messages, "create",
                          side_effect=[initial, synthesis]) as mock_create:
            generator.generate_response("q", tools=FAKE_TOOLS, tool_manager=tool_manager)

        synthesis_kwargs = mock_create.call_args_list[1][1]
        assert "tools" not in synthesis_kwargs

    def test_tool_use_without_tool_manager_returns_string(self, generator):
        """
        When stop_reason="tool_use" but tool_manager=None, the code must not
        crash with AttributeError trying to access .text on a ToolUseBlock.
        Instead it should return an empty string (or any text block it finds).

        We simulate a real ToolUseBlock by using spec to remove .text attribute.
        """
        block = MagicMock(spec=["type", "name", "input", "id"])
        block.type = "tool_use"
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = [block]

        with patch.object(generator.client.messages, "create", return_value=response):
            result = generator.generate_response("test", tools=FAKE_TOOLS, tool_manager=None)

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Group E — System prompt regression
# ---------------------------------------------------------------------------

class TestSystemPrompt:

    def test_one_tool_limit_removed_from_system_prompt(self):
        """
        The old "One tool call per query maximum" instruction contradicts the
        new 2-round behaviour and must no longer appear in the prompt.
        """
        assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT, (
            "The system prompt still contains the old single-tool restriction. "
            "Remove it so Claude knows it can chain a second tool call."
        )

    def test_system_prompt_describes_sequential_calls(self):
        """The prompt must tell Claude it can make 2 sequential tool calls."""
        prompt = AIGenerator.SYSTEM_PROMPT.lower()
        assert "2" in prompt or "two" in prompt, (
            "System prompt should mention that 2 (or two) sequential rounds are allowed."
        )
        assert "sequential" in prompt or "second" in prompt or "chain" in prompt, (
            "System prompt should describe the sequential / chaining capability."
        )
