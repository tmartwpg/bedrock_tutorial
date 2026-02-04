"""
LEVEL 5: Tool Use & Function Calling

Building on Level 4, we now enable the model to:
1. Call functions/tools that we define
2. Process tool results
3. Continue reasoning based on tool outputs
4. Build simple agents

Topics covered:
  - Tool/function definitions
  - Tool use in Bedrock API
  - Processing tool responses
  - Agent loops
  - Real-world use cases

Expected output:
  Model reasoning and using tools to accomplish tasks
"""

import json
import math
from typing import Any
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, ToolMessage
from utils import pretty_print
from config import MODEL_ID, SAMPLE_TOOLS


# ============================================================================
# Simple Tool Implementations
# ============================================================================

def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Simulate getting weather for a location.

    Args:
        location: City and state
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information as string
    """
    # In a real app, this would call a weather API
    # For this tutorial, we'll return simulated data
    weather_data = {
        "San Francisco, CA": {"temp": 18, "condition": "Cloudy"},
        "New York, NY": {"temp": 5, "condition": "Snowy"},
        "Miami, FL": {"temp": 28, "condition": "Sunny"},
        "Seattle, WA": {"temp": 12, "condition": "Rainy"},
    }

    if location in weather_data:
        data = weather_data[location]
        temp = data["temp"]
        # Convert to Fahrenheit if requested
        if unit == "fahrenheit":
            temp = (temp * 9/5) + 32
        return f"Weather in {location}: {data['condition']}, {temp}° {unit}"
    else:
        return f"Weather data not available for {location}"


def calculate(operation: str, a: float, b: float) -> float:
    """
    Perform a mathematical calculation.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        Result of the operation
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None,
    }

    if operation not in operations:
        return f"Unknown operation: {operation}"

    result = operations[operation](a, b)
    if result is None:
        return "Cannot divide by zero"
    return result


# Dictionary mapping tool names to their implementations
TOOLS_IMPLEMENTATION = {
    "get_weather": get_weather,
    "calculate": calculate,
}


class ToolUseTutorial:
    """
    Demonstrates tool use and function calling with Bedrock.

    The model can decide to call tools, and we process those calls
    to accomplish complex tasks.
    """

    def __init__(self):
        """Initialize the LangChain Bedrock model."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )

        # Bind tools to the model
        # This tells the model what tools are available
        self.model_with_tools = self.model.bind_tools(
            SAMPLE_TOOLS,
            tool_choice="auto"  # Model decides when to use tools
        )

    def process_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool call and return the result.

        Args:
            tool_name: Name of the tool to call
            tool_input: Arguments for the tool

        Returns:
            Result of the tool call as string
        """
        if tool_name not in TOOLS_IMPLEMENTATION:
            return f"Tool not found: {tool_name}"

        try:
            tool_func = TOOLS_IMPLEMENTATION[tool_name]
            # Call the tool with the provided arguments
            result = tool_func(**tool_input)
            return str(result)
        except Exception as e:
            return f"Error calling tool: {str(e)}"

    def example_1_weather_query(self):
        """Example 1: Query weather for different locations."""
        pretty_print("Example 1: Weather Query with Tool Use", "")

        prompt = "What's the weather like in San Francisco, CA and Miami, FL?"

        print(f"User: {prompt}\n")

        # Send request to model
        messages = [HumanMessage(content=prompt)]
        response = self.model_with_tools.invoke(messages)

        # Keep looping until model stops making tool calls
        while response.tool_calls:
            print("Model decides to use tools:")

            # Add the assistant's response with tool calls
            messages.append(response)

            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"  • Calling: {tool_name} with {tool_args}")

                result = self.process_tool_call(tool_name, tool_args)
                print(f"    Result: {result}")

                # Add tool result to messages
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

            # Get next response from model
            response = self.model_with_tools.invoke(messages)

        # Print final response (when model stops using tools)
        print(f"\nAssistant: {response.content}\n")

    def example_2_calculation_chain(self):
        """Example 2: Multi-step calculation."""
        pretty_print("Example 2: Multi-Step Calculation", "")

        prompt = "Calculate: (15 + 7) * 3, then divide the result by 2"

        print(f"User: {prompt}\n")

        messages = [HumanMessage(content=prompt)]
        response = self.model_with_tools.invoke(messages)

        step_count = 0
        while response.tool_calls:
            step_count += 1
            print(f"Step {step_count}:")

            # Add the assistant's response with tool calls
            messages.append(response)

            # Process each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"  Calling: {tool_name}({tool_args})")

                result = self.process_tool_call(tool_name, tool_args)
                print(f"  Result: {result}\n")

                # Add tool result to messages
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

            # Get next response
            response = self.model_with_tools.invoke(messages)

        print(f"Final Answer: {response.content}\n")

    def example_3_mixed_reasoning(self):
        """Example 3: Model uses tools and reasoning together."""
        pretty_print("Example 3: Tool Use with Reasoning", "")

        prompt = """It's currently spring in San Francisco.
        What's the weather like there?
        Based on that, suggest some activities."""

        print(f"User: {prompt}\n")

        messages = [HumanMessage(content=prompt)]
        response = self.model_with_tools.invoke(messages)

        # Process tool calls if any
        if response.tool_calls:
            print("Model uses tools to get information:")
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                print(f"  • {tool_name} with {tool_args}")

                result = self.process_tool_call(tool_name, tool_args)
                print(f"    -> {result}")

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

            # Get final response with reasoning
            final_response = self.model_with_tools.invoke(messages)
            print(f"\nAssistant reasoning:\n{final_response.content}\n")
        else:
            print(f"Assistant: {response.content}\n")

    def example_4_error_handling(self):
        """Example 4: Handle invalid tool calls gracefully."""
        pretty_print("Example 4: Error Handling in Tool Use", "")

        prompt = "Please divide 100 by 0 and tell me what happens"

        print(f"User: {prompt}\n")

        messages = [HumanMessage(content=prompt)]
        response = self.model_with_tools.invoke(messages)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                print(f"Model calls: {tool_call['name']} with {tool_call['args']}")

                # This will return an error message for divide by zero
                result = self.process_tool_call(
                    tool_call['name'],
                    tool_call['args']
                )
                print(f"Tool result: {result}")

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )

            final_response = self.model_with_tools.invoke(messages)
            print(f"\nAssistant handles error:\n{final_response.content}\n")

    def example_5_no_tool_needed(self):
        """Example 5: Model decides NOT to use tools."""
        pretty_print("Example 5: When Model Doesn't Need Tools", "")

        prompt = "What is the capital of France?"

        print(f"User: {prompt}\n")

        messages = [HumanMessage(content=prompt)]
        response = self.model_with_tools.invoke(messages)

        if response.tool_calls:
            print("Model decided to use tools")
        else:
            print("Model answers directly without tools:")
            print(f"Assistant: {response.content}\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 5: Tool Use & Function Calling")
        print("=" * 80)

        try:
            self.example_1_weather_query()
            self.example_2_calculation_chain()
            self.example_3_mixed_reasoning()
            self.example_4_error_handling()
            self.example_5_no_tool_needed()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. Tools/functions extend model capabilities")
            print("  2. Models decide when to use tools (if tool_choice='auto')")
            print("  3. Tool results are fed back to model for reasoning")
            print("  4. Can chain multiple tool calls")
            print("  5. Error handling is important for robustness")
            print("  6. Models can answer questions without tools too")
            print("  7. This enables agentic behavior")

        except Exception as e:
            pretty_print("ERROR", str(e))
            import traceback
            traceback.print_exc()


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates tool use and function calling.

    Usage:
        python level_05_tool_use.py
    """

    tutorial = ToolUseTutorial()
    tutorial.demonstrate()
