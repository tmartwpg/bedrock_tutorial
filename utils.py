"""
Shared utilities for the Bedrock tutorial.

This module contains helper functions used across multiple levels,
including client initialization, formatting, and common operations.
"""

import json
from typing import Any
import boto3
from config import AWS_REGION, MODEL_ID


def get_bedrock_client():
    """
    Initialize and return a boto3 Bedrock Runtime client.

    Returns:
        boto3 bedrock-runtime client configured for the tutorial region
    """
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_bedrock_embeddings_client():
    """
    Initialize and return a boto3 Bedrock client for embeddings.

    Returns:
        boto3 bedrock-runtime client for embedding models
    """
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def pretty_print(title: str, content: Any, width: int = 80) -> None:
    """
    Pretty print content with a formatted title.

    Args:
        title: The section title
        content: Content to print (will be formatted as JSON if dict)
        width: Width of the separator line
    """
    # Print separator line
    print("\n" + "=" * width)
    # Print title
    print(f"  {title}")
    print("=" * width)

    # Format and print content
    if isinstance(content, dict):
        print(json.dumps(content, indent=2, default=str))
    elif isinstance(content, str):
        print(content)
    else:
        print(str(content))


def format_messages(system: str, user: str) -> list:
    """
    Format system and user messages into Claude message format.

    Args:
        system: System prompt (instructions/context)
        user: User message/query

    Returns:
        Formatted messages list for Bedrock API
    """
    return [
        {"role": "user", "content": user}
    ], system


def extract_response_text(response: dict) -> str:
    """
    Extract the text content from a Bedrock API response.

    Args:
        response: Raw response from Bedrock invoke_model

    Returns:
        Extracted text content
    """
    # The response body is a stream - read it
    response_body = response.get("body", {})
    if hasattr(response_body, "read"):
        # It's a stream
        response_text = response_body.read().decode("utf-8")
        response_dict = json.loads(response_text)
    else:
        response_dict = response_body

    # Extract the text from the response
    # Bedrock responses have this structure: {"content": [{"text": "..."}]}
    if isinstance(response_dict, str):
        response_dict = json.loads(response_dict)

    content = response_dict.get("content", [])
    if content and len(content) > 0:
        return content[0].get("text", "")

    return ""


def format_tokens_used(response: dict) -> dict:
    """
    Extract token usage information from response.

    Args:
        response: Response from Bedrock invoke_model

    Returns:
        Dictionary with input_tokens and output_tokens
    """
    # Response metadata varies, but usually contains usage info
    if "usage" in response:
        return response["usage"]
    return {"input_tokens": 0, "output_tokens": 0}


def invoke_model_simple(
    client: Any,
    user_message: str,
    system_message: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> str:
    """
    Simple helper to invoke the model and get text response.

    This is used across multiple levels to reduce code duplication.

    Args:
        client: Bedrock runtime client
        user_message: The user's message/prompt
        system_message: Optional system prompt for context
        temperature: Model temperature (creativity) 0.0-1.0
        max_tokens: Maximum tokens in response

    Returns:
        The model's response as a string
    """
    # Build the messages list
    messages = [{"role": "user", "content": user_message}]

    # Build the request body
    request_body = {
        "model": MODEL_ID,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    # Add system prompt if provided
    if system_message:
        request_body["system"] = system_message

    # Invoke the model
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(request_body)
    )

    # Extract and return the text
    return extract_response_text(response)


# ============================================================================
# Conversation History Management
# ============================================================================

class ConversationBuffer:
    """
    Simple conversation history buffer for multi-turn conversations.

    Maintains conversation state and manages message history.
    """

    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation buffer.

        Args:
            max_turns: Maximum number of turns to keep in history
        """
        self.messages = []
        self.max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def get_messages(self) -> list:
        """Get all messages in conversation."""
        return self.messages

    def get_context(self) -> str:
        """Get formatted conversation context for display."""
        context = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            context.append(f"{role}:\n{content}\n")
        return "\n".join(context)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def _trim_history(self) -> None:
        """Keep only recent messages if exceeding max_turns."""
        # Each turn is a user message + assistant response (2 messages)
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            # Keep the oldest system context if any, then trim to max
            self.messages = self.messages[-max_messages:]


# ============================================================================
# JSON/Structured Output Helpers
# ============================================================================

def parse_json_from_response(response_text: str) -> dict:
    """
    Extract and parse JSON from a response that might contain extra text.

    Args:
        response_text: Response text that may contain JSON

    Returns:
        Parsed JSON as dictionary
    """
    # Try parsing the whole response first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # If that fails, try to find JSON in the response
    # Look for JSON wrapped in markdown code blocks
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        json_str = response_text[start:end].strip()
        return json.loads(json_str)

    # Try finding { ... }
    if "{" in response_text and "}" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
        return json.loads(json_str)

    raise ValueError(f"Could not find valid JSON in response: {response_text}")
