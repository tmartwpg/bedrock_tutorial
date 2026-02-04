"""
LEVEL 7: LangChain Chains - Composing Operations

Building on Level 6, we now use LangChain chains to:
1. Compose multiple operations into workflows
2. Chain prompts together (output of one is input to another)
3. Use built-in chains for common patterns
4. Create custom chains
5. Handle complex workflows

Topics covered:
  - LangChain chains
  - Prompt + LLM chains
  - Sequential chains
  - LCEL (LangChain Expression Language)
  - Composable workflows

Expected output:
  Complex workflows built from simple composable pieces
"""

import json
from typing import List, Any
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from utils import pretty_print
from config import MODEL_ID


class ChainsTutorial:
    """
    Demonstrates LangChain chains for composing operations.

    Chains let us build complex workflows from simple components.
    """

    def __init__(self):
        """Initialize the model."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )

    def example_1_basic_chain(self):
        """Example 1: Simple chain - Prompt + Model."""
        pretty_print("Example 1: Basic Prompt + Model Chain", "")

        # Step 1: Create a prompt template
        prompt = PromptTemplate(
            input_variables=["topic", "style"],
            template="""Write a {style} paragraph about {topic}.
Keep it to 3-4 sentences."""
        )

        # Step 2: Create a chain using LangChain expression language (LCEL)
        # The | operator chains components together
        chain = prompt | self.model

        # Step 3: Run the chain
        result = chain.invoke({
            "topic": "renewable energy",
            "style": "academic"
        })

        print(f"Output:\n{result.content}\n")
        print("[OK] Basic chain created using | operator!\n")

    def example_2_multi_step_chain(self):
        """Example 2: Multi-step chain with output parsing."""
        pretty_print("Example 2: Multi-Step Chain with Parsing", "")

        # Step 1: Create prompt to generate ideas - be explicit about the format
        idea_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Generate EXACTLY 5 creative ideas for a project about {topic}. Format: idea1, idea2, idea3, idea4, idea5"
        )

        # Step 2: Create output parser for comma-separated list
        parser = CommaSeparatedListOutputParser()

        # Step 3: Create chain
        chain = idea_prompt | self.model | parser

        # Step 4: Run the chain
        result = chain.invoke({"topic": "sustainable cities"})

        # Step 5: Limit to 5 ideas as a safety measure
        result = result[:5]

        print(f"Generated {len(result)} ideas:")
        for i, idea in enumerate(result, 1):
            print(f"  {i}. {idea.strip()}")
        print()

    def example_3_sequential_chain(self):
        """Example 3: Two sequential operations."""
        pretty_print("Example 3: Sequential Operations", "")

        # Step 1: Generate a topic
        topic_prompt = PromptTemplate(
            input_variables=[],
            template="Suggest one interesting technology topic"
        )

        topic_chain = topic_prompt | self.model

        print("Step 1: Generating a topic...")
        topic_result = topic_chain.invoke({})
        topic = topic_result.content

        print(f"Generated topic: {topic}\n")

        # Step 2: Summarize the topic
        summary_prompt = PromptTemplate(
            input_variables=["topic"],
            template="Summarize this topic in 2 sentences: {topic}"
        )

        summary_chain = summary_prompt | self.model

        print("Step 2: Creating summary...")
        summary_result = summary_chain.invoke({"topic": topic})

        print(f"Summary:\n{summary_result.content}\n")
        print("[OK] Sequential operations completed!\n")

    def example_4_branching_workflow(self):
        """Example 4: Chain with branching logic."""
        pretty_print("Example 4: Conditional Chain Logic", "")

        text = """Artificial Intelligence has revolutionized many industries.
        From healthcare to finance, AI systems are making a significant impact."""

        print(f"Input text:\n{text}\n")

        # Analyze sentiment first
        sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Analyze the sentiment of this text.
Answer only with: positive, negative, or neutral.

Text: {text}"""
        )

        sentiment_chain = sentiment_prompt | self.model

        sentiment_result = sentiment_chain.invoke({"text": text})
        sentiment = sentiment_result.content.strip().lower()

        print(f"Sentiment analysis: {sentiment}")

        # Branch based on sentiment
        if "positive" in sentiment:
            followup_prompt = PromptTemplate(
                input_variables=["text"],
                template="This text is positive. Expand on its positive aspects: {text}"
            )
        elif "negative" in sentiment:
            followup_prompt = PromptTemplate(
                input_variables=["text"],
                template="This text is negative. Provide counterarguments: {text}"
            )
        else:
            followup_prompt = PromptTemplate(
                input_variables=["text"],
                template="Provide a balanced view of: {text}"
            )

        followup_chain = followup_prompt | self.model
        followup_result = followup_chain.invoke({"text": text})

        print(f"\nFollowup response:\n{followup_result.content}\n")
        print("[OK] Branching logic allows dynamic workflows!\n")

    def example_5_complex_workflow(self):
        """Example 5: Complex multi-step workflow."""
        pretty_print("Example 5: Complex Multi-Step Workflow", "")

        # Workflow: Analyze requirement -> Generate solution -> Create test cases

        requirement = "Create a function to validate email addresses"

        print(f"Requirement: {requirement}\n")

        # Step 1: Break down requirements
        print("Step 1: Analyzing requirements...")
        analysis_prompt = PromptTemplate(
            input_variables=["requirement"],
            template="""Analyze this requirement and list 3 key concerns:
{requirement}

Format: Just list the concerns, one per line."""
        )

        analysis_chain = analysis_prompt | self.model
        analysis_result = analysis_chain.invoke({"requirement": requirement})
        print(f"Analysis:\n{analysis_result.content}\n")

        # Step 2: Generate solution
        print("Step 2: Generating solution...")
        solution_prompt = PromptTemplate(
            input_variables=["requirement"],
            template="""Provide a Python solution for: {requirement}

Include:
- Function definition
- One example usage
- Brief comment explaining the approach

Keep it concise."""
        )

        solution_chain = solution_prompt | self.model
        solution_result = solution_chain.invoke({"requirement": requirement})
        solution = solution_result.content
        print(f"Solution (first 300 chars):\n{solution[:300]}...\n")

        # Step 3: Generate tests
        print("Step 3: Generating test cases...")
        test_prompt = PromptTemplate(
            input_variables=["requirement"],
            template="""Create 3 test cases for: {requirement}

Format as a list. Include:
- Valid case
- Invalid case
- Edge case"""
        )

        test_chain = test_prompt | self.model
        test_result = test_chain.invoke({"requirement": requirement})
        print(f"Test cases:\n{test_result.content}\n")

        print("[OK] Complex workflow completed through chaining!\n")

    def example_6_chain_with_retry(self):
        """Example 6: Chain with implicit retry in workflow."""
        pretty_print("Example 6: Robust Chaining", "")

        # Create a chain that processes user input
        validation_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""Validate if this is a valid user input for a survey.
Input: {user_input}

Respond with ONLY: valid or invalid"""
        )

        processing_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="Process and summarize this input: {user_input}"
        )

        # Try processing some inputs
        inputs = [
            "I really enjoy hiking and outdoor activities",
            "xyz123!@#",
            "This is a legitimate survey response about my preferences"
        ]

        for user_input in inputs:
            print(f"Input: {user_input}")

            # Check validity
            validation_chain = validation_prompt | self.model
            validation = validation_chain.invoke({"user_input": user_input})

            if "valid" in validation.content.lower():
                # Process if valid
                processing_chain = processing_prompt | self.model
                result = processing_chain.invoke({"user_input": user_input})
                print(f"[OK] Processed: {result.content[:100]}...")
            else:
                print(f"[FAIL] Invalid input, skipped")

            print()

        print("[OK] Chains can include validation and error handling!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 7: LangChain Chains")
        print("=" * 80)

        try:
            self.example_1_basic_chain()
            self.example_2_multi_step_chain()
            self.example_3_sequential_chain()
            self.example_4_branching_workflow()
            self.example_5_complex_workflow()
            self.example_6_chain_with_retry()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. Chains compose operations using | operator (LCEL)")
            print("  2. Prompt + Model is the simplest chain")
            print("  3. Chains can include output parsers")
            print("  4. Sequential chains run steps in order")
            print("  5. Branching logic creates conditional workflows")
            print("  6. Complex workflows built from simple pieces")
            print("  7. Chains are reusable and composable")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates LangChain chains.

    Usage:
        python level_07_chains.py
    """

    tutorial = ChainsTutorial()
    tutorial.demonstrate()
