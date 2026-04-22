import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- You may make up to 2 sequential tool calls when needed (one per API round)
- Make one tool call at a time — do not call multiple tools simultaneously
- Use a second tool call only if the first result reveals you need more information
- Use `get_course_outline` for questions about a course's structure, outline, syllabus, or lesson list
- Use `search_course_content` for questions about specific content, concepts, or details within lessons
- Synthesize all tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course outline questions**: Use `get_course_outline`, then return the course title as a heading, the course link, and the full numbered lesson list
- **Course-specific content questions**: Use `search_course_content`, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
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
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Safety: if tool_use fired but no tool_manager, find any text block rather than crashing
        if response.stop_reason == "tool_use":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Return direct response
        return response.content[0].text

    # Maximum number of sequential tool-calling rounds per query
    MAX_TOOL_ROUNDS = 2

    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Execute up to MAX_TOOL_ROUNDS sequential tool-calling rounds.

        Round N: execute tools → inject results → call API with tools still present.
        If Claude answers with text during an intermediate call, return immediately.
        After all rounds (or on tool error), make one final synthesis call without tools.
        """
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})
        current_response = initial_response

        for round_num in range(self.MAX_TOOL_ROUNDS):

            # Execute every tool_use block in the current response
            tool_results = []
            error_occurred = False
            for block in current_response.content:
                if block.type == "tool_use":
                    try:
                        result = tool_manager.execute_tool(block.name, **block.input)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        error_occurred = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                    if error_occurred:
                        break

            messages.append({"role": "user", "content": tool_results})

            # After the last allowed round or on error, fall through to synthesis
            if round_num == self.MAX_TOOL_ROUNDS - 1 or error_occurred:
                break

            # Intermediate call: tools still present so Claude can chain a second call
            next_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": base_params["tools"],
                "tool_choice": {"type": "auto"},
            }
            next_response = self.client.messages.create(**next_params)

            # Claude answered with text — no synthesis call needed
            if next_response.stop_reason != "tool_use":
                return next_response.content[0].text

            messages.append({"role": "assistant", "content": next_response.content})
            current_response = next_response

        # Final synthesis call — tools removed so Claude writes the answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text