"""
LEVEL 3: Prompt Templates - Reusable Prompt Patterns

Building on Level 2, we now use LangChain to:
1. Create reusable prompt templates
2. Inject variables into prompts
3. Build more complex prompt structures
4. Avoid string concatenation mess

Topics covered:
  - LangChain PromptTemplate
  - Variable substitution
  - Multi-variable templates
  - Template composition

Expected output:
  Responses from various prompt templates with injected variables
"""

import json
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from utils import pretty_print
from config import MODEL_ID


class PromptTemplateTutorial:
    """
    Demonstrates LangChain prompt templates for reusable, composable prompts.

    Templates are much better than string concatenation - they're readable,
    testable, and easy to maintain.
    """

    def __init__(self):
        """Initialize the LangChain Bedrock chat model."""
        # LangChain provides a nicer interface to Bedrock
        # ChatBedrock handles all the model invocation details
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )

    def example_1_simple_template(self):
        """Example 1: Simple template with one variable."""
        pretty_print("Example 1: Simple Template", "")

        # Create a prompt template with a variable
        # The {topic} will be replaced with our input
        template = PromptTemplate(
            input_variables=["topic"],
            template="Explain {topic} in one paragraph for beginners."
        )

        # Format the template with our topic
        topic = "photosynthesis"
        prompt = template.format(topic=topic)

        print(f"Template: {template.template}")
        print(f"Input variables: {template.input_variables}\n")
        print(f"Formatted prompt:\n{prompt}\n")

        # Use the formatted prompt with the model
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        print(f"Response:\n{response.content}\n")
        print("[OK] Template saved us from string concatenation!\n")

    def example_2_multi_variable_template(self):
        """Example 2: Template with multiple variables."""
        pretty_print("Example 2: Multi-Variable Template", "")

        # Template with multiple variables
        template = PromptTemplate(
            input_variables=["role", "topic", "audience"],
            template="""You are a {role}.
Explain {topic} to a {audience}.
Use simple language and examples."""
        )

        # Format with different values
        role = "kindergarten teacher"
        topic = "gravity"
        audience = "5-year-old child"

        prompt = template.format(role=role, topic=topic, audience=audience)

        print(f"Template variables: {template.input_variables}\n")
        print(f"Formatted prompt:\n{prompt}\n")

        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        print(f"Response:\n{response.content}\n")
        print("[OK] Multiple variables make templates flexible and reusable!\n")

    def example_3_question_answering_template(self):
        """Example 3: Reusable Q&A template."""
        pretty_print("Example 3: Question Answering Template", "")

        # Template for Q&A with context
        qa_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context: {context}

Question: {question}

Answer based on the context above."""
        )

        # Use the template multiple times with different inputs
        examples = [
            {
                "context": "Python is a high-level programming language created by Guido van Rossum in 1991.",
                "question": "When was Python created?"
            },
            {
                "context": "Mount Everest is 8,849 meters tall and located in the Himalayas.",
                "question": "How tall is Mount Everest?"
            }
        ]

        for i, example in enumerate(examples):
            prompt = qa_template.format(
                context=example["context"],
                question=example["question"]
            )

            print(f"Question {i+1}: {example['question']}")
            messages = [HumanMessage(content=prompt)]
            response = self.model.invoke(messages)
            print(f"Answer: {response.content}\n")

        print("[OK] Same template, different data - that's the power of templates!\n")

    def example_4_analysis_template_chain(self):
        """Example 4: Sequential templates (simple chain)."""
        pretty_print("Example 4: Template Chaining", "")

        # Step 1: Summarize template
        summarize_template = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text in 1-2 sentences:\n\n{text}"
        )

        # Step 2: Generate questions from summary
        question_template = PromptTemplate(
            input_variables=["summary"],
            template="Generate 3 questions that a reader might ask about this summary:\n\n{summary}"
        )

        # Sample text to process
        text = """Artificial Intelligence is transforming many industries. Machine learning algorithms
can now process vast amounts of data and find patterns that humans might miss. AI is being used
in healthcare for diagnosis, in finance for fraud detection, and in transportation for autonomous vehicles."""

        print("Original text:")
        print(f"{text}\n")

        # Step 1: Summarize
        summarize_prompt = summarize_template.format(text=text)
        messages = [HumanMessage(content=summarize_prompt)]
        summary_response = self.model.invoke(messages)
        summary = summary_response.content

        print(f"Summary:\n{summary}\n")

        # Step 2: Generate questions from the summary
        question_prompt = question_template.format(summary=summary)
        messages = [HumanMessage(content=question_prompt)]
        questions_response = self.model.invoke(messages)

        print(f"Generated Questions:\n{questions_response.content}\n")
        print("[OK] Templates can be chained together for complex workflows!\n")

    def example_5_system_and_user_template(self):
        """Example 5: Templates with system prompts."""
        pretty_print("Example 5: System + User Templates", "")

        # System prompt template
        system_template = """You are an expert {expert_type}.
Your goal is to provide {goal}.
Always use clear, professional language."""

        # User message template
        user_template = "I need help with: {user_question}"

        # Create prompt templates
        system_prompt = PromptTemplate(
            input_variables=["expert_type", "goal"],
            template=system_template
        )

        user_prompt = PromptTemplate(
            input_variables=["user_question"],
            template=user_template
        )

        # Format with values
        expert_type = "data scientist"
        goal = "accurate analysis and insights"
        user_question = "How do I handle missing data in my dataset?"

        system_msg = system_prompt.format(expert_type=expert_type, goal=goal)
        user_msg = user_prompt.format(user_question=user_question)

        print(f"System Template:\n{system_template}\n")
        print(f"User Template:\n{user_template}\n")
        print(f"Formatted System:\n{system_msg}\n")
        print(f"Formatted User:\n{user_msg}\n")

        # Send to model with system message
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        response = self.model.invoke(messages)

        print(f"Response:\n{response.content}\n")
        print("[OK] System and user templates work together for powerful prompts!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 3: Prompt Templates")
        print("=" * 80)

        try:
            self.example_1_simple_template()
            self.example_2_multi_variable_template()
            self.example_3_question_answering_template()
            self.example_4_analysis_template_chain()
            self.example_5_system_and_user_template()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. PromptTemplate creates reusable prompt patterns")
            print("  2. Variables are injected using format()")
            print("  3. Templates can be chained together")
            print("  4. Much cleaner than string concatenation")
            print("  5. Easy to version control and test")
            print("  6. LangChain provides the ChatBedrock interface")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates prompt templates.

    Usage:
        python level_03_prompt_templates.py
    """

    tutorial = PromptTemplateTutorial()
    tutorial.demonstrate()
