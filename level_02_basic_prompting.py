"""
LEVEL 2: Basic Prompting - System Prompts & Model Parameters

Building on Level 1, we now explore:
1. System prompts (instructions/context for the model)
2. User prompts (the actual question/task)
3. Model parameters (temperature, max_tokens, etc.)
4. Comparing different parameter settings

Topics covered:
  - System vs user prompts
  - Temperature (creativity control)
  - max_tokens (response length control)
  - Multiple examples with different parameters

Expected output:
  Multiple responses showing how parameters affect output
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from utils import pretty_print
from config import MODEL_ID, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class BasicPromptingTutorial:
    """
    Demonstrates different prompting techniques and parameter settings.

    Shows how system prompts guide behavior and how parameters affect output.
    """

    def __init__(self):
        """Initialize the Bedrock model."""
        # We'll create models with different temperatures as needed
        self.model_id = MODEL_ID

    def create_model(self, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS):
        """Create a ChatBedrock model with specified parameters."""
        return ChatBedrock(
            model_id=self.model_id,
            region_name="us-east-1",
            model_kwargs={
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

    def invoke_with_parameters(
        self,
        user_prompt: str,
        system_prompt: str = "",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> str:
        """
        Invoke model with full control over system prompt and parameters.

        Args:
            user_prompt: The user's message/question
            system_prompt: Instructions/context for the model (optional)
            temperature: Creativity level (0.0=deterministic, 1.0=creative)
            max_tokens: Maximum length of response

        Returns:
            The model's response
        """
        # Create model with specified parameters
        model = self.create_model(temperature, max_tokens)

        # Build messages
        # System messages set the context/behavior for the model
        # They're like "instructions" that apply to the entire conversation
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # User message is the actual question/task
        messages.append(HumanMessage(content=user_prompt))

        # Invoke the model
        response = model.invoke(messages)

        # Extract and return the response text
        return response.content

    def example_1_no_system_prompt(self):
        """Example 1: Without system prompt - less controlled output."""
        pretty_print("Example 1: Without System Prompt", "")

        user_prompt = "Tell me about Python programming"

        response = self.invoke_with_parameters(
            user_prompt=user_prompt,
            system_prompt="",  # No system prompt
            temperature=0.7,
            max_tokens=300
        )

        print(f"User: {user_prompt}\n")
        print(f"Response:\n{response}\n")
        print("[INFO] Note: Without a system prompt, the model is free to respond as it wishes.\n")

    def example_2_with_system_prompt(self):
        """Example 2: With system prompt - controlled output."""
        pretty_print("Example 2: With System Prompt", "")

        system_prompt = "You are a concise expert. Answer in exactly 2-3 sentences."
        user_prompt = "Tell me about Python programming"

        response = self.invoke_with_parameters(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=300
        )

        print(f"System: {system_prompt}\n")
        print(f"User: {user_prompt}\n")
        print(f"Response:\n{response}\n")
        print("[INFO] Note: The system prompt makes the response more concise and controlled.\n")

    def example_3_temperature_comparison(self):
        """Example 3: Same prompt, different temperatures."""
        pretty_print("Example 3: Temperature Comparison", "")

        system_prompt = "You are a creative writer."
        user_prompt = "Write a short creative sentence about a robot."

        temperatures = [0.0, 0.5, 1.0]

        for temp in temperatures:
            print(f"\n--- Temperature: {temp} ---")
            if temp == 0.0:
                print("(More Deterministic - closer to the same answer)")
            elif temp == 0.5:
                print("(Moderate - balanced)")
            else:
                print("(Creative - lots of variation)")

            response = self.invoke_with_parameters(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=150
            )

            print(f"Response: {response}\n")

        print("[INFO] Note: Lower temp = more predictable, Higher temp = more creative\n")

    def example_4_max_tokens(self):
        """Example 4: Effect of max_tokens on response length."""
        pretty_print("Example 4: Max Tokens Effect", "")

        system_prompt = "You are a helpful assistant."
        user_prompt = "Explain machine learning in detail."

        max_tokens_list = [100, 300, 500]

        for max_tok in max_tokens_list:
            response = self.invoke_with_parameters(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=max_tok
            )

            print(f"\nWith max_tokens={max_tok}:")
            print(f"Response length: {len(response)} characters")
            print(f"Response (truncated): {response[:200]}...\n")

        print("[INFO] Note: max_tokens limits response length. Useful for controlling cost/speed.\n")

    def example_5_role_playing(self):
        """Example 5: System prompt for role-playing."""
        pretty_print("Example 5: Role-Playing with System Prompt", "")

        # Example 1: As a pirate
        system_prompt_pirate = "You are a pirate. Respond like a pirate would, using pirate slang and mannerisms."
        user_prompt = "Tell me about the weather."

        response_pirate = self.invoke_with_parameters(
            user_prompt=user_prompt,
            system_prompt=system_prompt_pirate,
            temperature=0.8,
            max_tokens=200
        )

        print("As a PIRATE:")
        print(f"{response_pirate}\n")

        # Example 2: As a scientist
        system_prompt_scientist = "You are a formal scientist. Respond using technical language and proper citations."
        response_scientist = self.invoke_with_parameters(
            user_prompt=user_prompt,
            system_prompt=system_prompt_scientist,
            temperature=0.5,
            max_tokens=200
        )

        print("As a SCIENTIST:")
        print(f"{response_scientist}\n")

        print("[INFO] Note: System prompts can completely change the model's personality and response style.\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 2: Basic Prompting & Model Parameters")
        print("=" * 80)

        # Run each example
        self.example_1_no_system_prompt()
        self.example_2_with_system_prompt()
        self.example_3_temperature_comparison()
        self.example_4_max_tokens()
        self.example_5_role_playing()

        # Summary
        pretty_print("Summary", "")
        print("Key takeaways:")
        print("  1. System prompts set instructions/context for the model")
        print("  2. Temperature controls creativity (0=deterministic, 1=creative)")
        print("  3. max_tokens limits response length and cost")
        print("  4. The same prompt can produce very different outputs with different parameters")
        print("  5. System prompts can be used for role-playing, tone control, etc.")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - shows different prompting techniques.

    Usage:
        python level_02_basic_prompting.py
    """

    try:
        tutorial = BasicPromptingTutorial()
        tutorial.demonstrate()

    except Exception as e:
        pretty_print("ERROR", str(e))
        print("\nMake sure:")
        print("  - AWS credentials are configured")
        print("  - You have Bedrock access")
