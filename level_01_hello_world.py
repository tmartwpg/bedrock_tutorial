"""
LEVEL 1: Hello World - Basic AWS Bedrock Connection

This is the foundation. We'll:
1. Connect to AWS Bedrock
2. Invoke the Claude model
3. Get and display the response

This demonstrates the simplest possible interaction with Bedrock.

Topics covered:
  - AWS Bedrock client initialization
  - Model invocation
  - Response parsing
  - Basic error handling

Expected output:
  A friendly "Hello World" response from Claude
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from utils import pretty_print
from config import MODEL_ID, AWS_REGION


class HelloWorldTutorial:
    """
    Simple class to demonstrate basic Bedrock connection and invocation.

    This is the most basic example - just get text in, get text out.
    We use LangChain's ChatBedrock which handles all the API details.
    """

    def __init__(self):
        """Initialize the Bedrock client through LangChain."""
        # Initialize the ChatBedrock model
        # LangChain handles all the complexity of API versions and formatting
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1"
        )

    def invoke_model(self, prompt: str) -> str:
        """
        Invoke Claude model with a simple prompt and return response.

        Args:
            prompt: The user's question or prompt

        Returns:
            The model's response text
        """
        # Step 1: Create a message object
        # LangChain uses HumanMessage to represent user input
        message = HumanMessage(content=prompt)

        # Step 2: Invoke the model
        # The model processes the message and returns a response
        response = self.model.invoke([message])

        # Step 3: Extract the text content
        # LangChain responses have a .content attribute with the text
        return response.content

    def demonstrate(self):
        """Run the Hello World demonstration."""
        # Our simple prompt
        prompt = "Say hello and explain briefly what AWS Bedrock is in 2 sentences."

        # Print what we're doing
        pretty_print("Level 1: Hello World", f"Prompt: {prompt}")

        # Invoke the model
        response = self.invoke_model(prompt)

        # Display the response
        pretty_print("Model Response", response)

        # Show what happened
        print("\n[OK] Success! You've made your first Bedrock API call.")
        print(f"  - Region: {AWS_REGION}")
        print(f"  - Model: {MODEL_ID}")
        print(f"  - Sent prompt: '{prompt}'")
        print(f"  - Got response: {len(response)} characters")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - run this file directly to see Hello World in action.

    Usage:
        python level_01_hello_world.py

    This will:
    1. Connect to AWS Bedrock
    2. Send a simple prompt
    3. Display the response
    """

    try:
        # Create the tutorial instance
        tutorial = HelloWorldTutorial()

        # Run the demonstration
        tutorial.demonstrate()

    except Exception as e:
        # If something goes wrong, print the error
        pretty_print("ERROR", str(e))
        print("\nTroubleshooting:")
        print("  - Ensure AWS credentials are configured: aws configure")
        print("  - Verify you have Bedrock access in the console")
        print("  - Check your AWS region in config.py")
