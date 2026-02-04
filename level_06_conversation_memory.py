"""
LEVEL 6: Conversation Memory - Multi-Turn Conversations

Building on Level 5, we now learn to:
1. Maintain conversation history across turns
2. Build context-aware multi-turn dialogs
3. Manage memory efficiently (don't overflow context)
4. Implement conversation buffers
5. Summarize long conversations

Topics covered:
  - Conversation history management
  - Context windows and token limits
  - Memory patterns (buffer, summary, sliding window)
  - Stateful conversations
  - Context retrieval

Expected output:
  Multi-turn conversations that remember previous context
"""

import json
from typing import List
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import pretty_print, ConversationBuffer
from config import MODEL_ID, MODEL_PRICING


class ConversationMemoryTutorial:
    """
    Demonstrates managing conversation memory for multi-turn dialogs.

    The key insight: LLMs don't inherently remember context across calls.
    We must explicitly pass conversation history each time.
    """

    def __init__(self):
        """Initialize the model."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )
        # Track API usage
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def _track_usage(self, response):
        """Track API call and token usage from response."""
        self.api_calls += 1

        # Extract token usage if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self.input_tokens += response.usage_metadata.get('input_tokens', 0)
            self.output_tokens += response.usage_metadata.get('output_tokens', 0)

    def example_1_without_memory(self):
        """Example 1: Conversation WITHOUT memory - loses context."""
        pretty_print("Example 1: Without Memory (loses context)", "")

        # Each call is independent - model doesn't remember previous messages
        print("User: My name is Alice and I like Python programming.")
        response1 = self.model.invoke([
            HumanMessage(content="My name is Alice and I like Python programming.")
        ])
        self._track_usage(response1)
        print(f"Assistant: {response1.content}\n")

        # Second turn - model has no context
        print("User: What's my name?")
        response2 = self.model.invoke([
            HumanMessage(content="What's my name?")  # Model doesn't know!
        ])
        self._track_usage(response2)
        print(f"Assistant: {response2.content}")
        print("[ERROR] Model forgot the name because we didn't pass history!\n")

    def example_2_with_memory(self):
        """Example 2: Conversation WITH memory - preserves context."""
        pretty_print("Example 2: With Memory (preserves context)", "")

        # Create conversation history
        messages = []

        # Turn 1
        user_input_1 = "My name is Bob and I work as a data scientist."
        print(f"User: {user_input_1}")
        messages.append(HumanMessage(content=user_input_1))

        response1 = self.model.invoke(messages)
        self._track_usage(response1)
        print(f"Assistant: {response1.content}\n")
        messages.append(AIMessage(content=response1.content))

        # Turn 2 - include all previous messages
        user_input_2 = "What was my name and job again?"
        print(f"User: {user_input_2}")
        messages.append(HumanMessage(content=user_input_2))

        # Pass entire history - model now has context
        response2 = self.model.invoke(messages)
        self._track_usage(response2)
        print(f"Assistant: {response2.content}\n")
        messages.append(AIMessage(content=response2.content))

        # Turn 3 - another question
        user_input_3 = "I'm interested in machine learning. What should I study?"
        print(f"User: {user_input_3}")
        messages.append(HumanMessage(content=user_input_3))

        response3 = self.model.invoke(messages)
        self._track_usage(response3)
        print(f"Assistant: {response3.content}")
        print("[OK] Model remembers context by including full message history!\n")

    def example_3_conversation_buffer(self):
        """Example 3: Using ConversationBuffer helper.

        ConversationBuffer is a helper class that automatically keeps track of your
        chat history. Instead of manually managing a list of messages, the buffer
        does it for you—add messages in, get the full history out when you need it."""
        
        pretty_print("Example 3: Using ConversationBuffer Class", "")

        # Create buffer from utils
        buffer = ConversationBuffer(max_turns=10)

        # Simulate a multi-turn conversation
        conversations = [
            ("What's your favorite programming language?", "My favorite is Python"),
            ("Why Python?", "It's clean, readable, and has great libraries"),
            ("What libraries do you like?", "I like NumPy, Pandas, and FastAPI"),
        ]

        for user_msg, _ in conversations:
            print(f"User: {user_msg}")
            
            # Stores in buffer for history tracking
            buffer.add_user_message(user_msg) 

            # Get response from model using buffer
            response = self.model.invoke(buffer.get_messages() + [ # Gets ALL previous messages from buffer
                HumanMessage(content=user_msg)
            ])
            self._track_usage(response)
            assistant_msg = response.content

            print(f"Assistant: {assistant_msg}\n")
            buffer.add_assistant_message(assistant_msg)

        # Show conversation history
        print("Conversation History:")
        print(buffer.get_context())
        print("[OK] Buffer manages conversation state automatically!\n")

    def example_4_with_system_prompt(self):
        """Example 4: System prompt with conversation memory."""
        pretty_print("Example 4: System Prompt with Memory", "")

        # Define a system prompt
        system_prompt = """You are a helpful coding tutor.
        - Answer questions about programming
        - Provide code examples when appropriate
        - Be encouraging and patient
        - Remember what the user has already told you"""

        messages = [SystemMessage(content=system_prompt)]

        # Conversation
        turns = [
            "I'm learning JavaScript. It's my first programming language.",
            "Can you give me a simple example?",
            "That's helpful! Now teach me about loops.",
        ]

        for user_input in turns:
            print(f"User: {user_input}")
            messages.append(HumanMessage(content=user_input))

            response = self.model.invoke(messages)
            self._track_usage(response)
            print(f"Assistant: {response.content}\n")
            messages.append(AIMessage(content=response.content))

        print("[OK] System prompt persists across turns, guiding the model's behavior!\n")

    def example_5_memory_limits(self):
        """Example 5: Managing memory limits (token limits)."""
        pretty_print("Example 5: Memory Limits & Windowing", "")

        print("Context window considerations:")
        print("  - Claude models have large context windows")
        print("  - But keeping full history uses more tokens (costs more)")
        print("  - Strategies: summarize, keep only recent, or use sliding window\n")

        # Example: Keep only recent messages
        buffer = ConversationBuffer(max_turns=3)  # Only keep last 3 turns

        print("With max_turns=3, oldest messages are dropped:")
        for i in range(6):
            user_msg = f"Turn {i+1}: This is a test message"
            assistant_msg = f"Response to turn {i+1}"

            buffer.add_user_message(user_msg)
            buffer.add_assistant_message(assistant_msg)

            print(f"  Turn {i+1}: {len(buffer.get_messages())} messages in buffer")

        print(f"\nFinal buffer has only recent turns:")
        print(buffer.get_context())
        print("[OK] Sliding window limits memory usage!\n")

    def example_6_conversation_summary(self):
        """Example 6: Summarizing conversations for memory."""
        pretty_print("Example 6: Conversation Summarization", "")

        # Long conversation
        print("Scenario: User has had a long conversation and context is growing.\n")

        buffer = ConversationBuffer(max_turns=100)

        # Add some conversation turns
        conversation_data = [
            ("I'm building a web app", "That's great! What technology?"),
            ("Using Python and FastAPI", "Excellent choice! FastAPI is very fast"),
            ("I need help with authentication", "Let me help with that"),
            ("How do I implement JWT?", "Here's how to use JWT..."),
        ]

        for user_msg, assistant_msg in conversation_data:
            buffer.add_user_message(user_msg)
            buffer.add_assistant_message(assistant_msg)

        print("Current conversation:")
        print(buffer.get_context())

        # Create summary prompt
        summary_prompt = f"""Summarize this conversation in 2-3 sentences:

{buffer.get_context()}"""

        print("Creating summary for memory optimization...")
        summary_response = self.model.invoke([
            HumanMessage(content=summary_prompt)
        ])
        self._track_usage(summary_response)

        print(f"Summary: {summary_response.content}\n")
        print("[OK] Summarization helps maintain context while saving tokens!\n")

    def example_7_complex_dialog(self):
        """Example 7: Complex multi-turn dialog with context."""
        pretty_print("Example 7: Complex Multi-Turn Dialog", "")

        system_prompt = "You are a helpful AI assistant helping with planning."

        messages = [SystemMessage(content=system_prompt)]

        # Multi-turn dialog
        dialog = [
            "I'm planning a trip to Japan for 2 weeks.",
            "I have a budget of $3000.",
            "I like hiking and visiting temples.",
            "What cities should I visit?",
            "How many days in each city?",
            "Can you give me a day-by-day itinerary?",
        ]

        for i, user_input in enumerate(dialog, 1):
            print(f"\nTurn {i}:")
            print(f"User: {user_input}")

            messages.append(HumanMessage(content=user_input))
            response = self.model.invoke(messages)
            self._track_usage(response)

            print(f"Assistant: {response.content[:200]}...")  # Truncate for display
            messages.append(AIMessage(content=response.content))

        print(f"\n[OK] Complex dialog with {len(messages)} total messages managed!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 6: Conversation Memory")
        print("=" * 80)

        try:
            self.example_1_without_memory()
            self.example_2_with_memory()
            self.example_3_conversation_buffer()
            self.example_4_with_system_prompt()
            self.example_5_memory_limits()
            self.example_6_conversation_summary()
            self.example_7_complex_dialog()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. LLMs don't inherently remember - pass full history")
            print("  2. ConversationBuffer helps manage message history")
            print("  3. System prompts persist across turns")
            print("  4. Need to manage token usage (history = more tokens)")
            print("  5. Strategies: summarization, sliding window, truncation")
            print("  6. Stateful conversations enable complex interactions")
            print("  7. Context is power - more history = better reasoning")

            # API Usage Report
            pretty_print("API Usage Report", "")
            total_tokens = self.input_tokens + self.output_tokens

            # Get pricing for current model
            input_price, output_price = MODEL_PRICING.get(
                MODEL_ID,
                (3.00, 15.00)  # Default to Sonnet pricing
            )

            # Calculate costs
            input_cost = (self.input_tokens / 1_000_000) * input_price
            output_cost = (self.output_tokens / 1_000_000) * output_price
            total_cost = input_cost + output_cost

            # Extract model name for display
            model_name = MODEL_ID.split(".")[-1] if "." in MODEL_ID else MODEL_ID

            print(f"  - Total API calls: {self.api_calls}")
            print(f"  - Total input tokens: {self.input_tokens:,}")
            print(f"  - Total output tokens: {self.output_tokens:,}")
            print(f"  - Total tokens: {total_tokens:,}")
            print(f"\nEstimated cost ({model_name}):")
            print(f"  - Input: ${input_cost:.4f} (${input_price} per 1M tokens)")
            print(f"  - Output: ${output_cost:.4f} (${output_price} per 1M tokens)")
            print(f"  - Total: ${total_cost:.4f}\n")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates conversation memory.

    Usage:
        python level_06_conversation_memory.py
    """

    tutorial = ConversationMemoryTutorial()
    tutorial.demonstrate()
