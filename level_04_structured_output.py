"""
LEVEL 4: Structured Output with Pydantic

Building on Level 3, we now learn to:
1. Define data models with Pydantic
2. Get the model to output structured JSON
3. Automatically parse and validate responses
4. Use structured data in Python code

Topics covered:
  - Pydantic models
  - JSON schema generation
  - Structured prompts that enforce format
  - Type safety and validation
  - Error handling for invalid outputs

Expected output:
  Structured, validated data from model responses
"""

import json
from typing import List
from pydantic import BaseModel, Field, ValidationError
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from utils import pretty_print, parse_json_from_response
from config import MODEL_ID


# ============================================================================
# Define Pydantic Models - These enforce structure and validation
# ============================================================================

class Person(BaseModel):
    """Model representing a person with validation."""
    name: str = Field(..., description="Person's full name")
    age: int = Field(..., description="Person's age")
    occupation: str = Field(..., description="Person's job")


class Recipe(BaseModel):
    """Model for a cooking recipe."""
    name: str = Field(..., description="Recipe name")
    cuisine: str = Field(..., description="Cuisine type")
    servings: int = Field(..., description="Number of servings")
    ingredients: List[str] = Field(..., description="List of ingredients")
    instructions: List[str] = Field(..., description="Step-by-step instructions")
    cooking_time_minutes: int = Field(..., description="Total cooking time in minutes")


class CodeReview(BaseModel):
    """Model for code review feedback."""
    file_name: str = Field(..., description="Name of the file reviewed")
    issues: List[str] = Field(..., description="List of issues found")
    improvements: List[str] = Field(..., description="Suggested improvements")
    rating: int = Field(..., description="Code quality rating 1-10", ge=1, le=10)
    overall_feedback: str = Field(..., description="Summary feedback")


class NewsArticle(BaseModel):
    """Model for a news article."""
    headline: str = Field(..., description="Article headline")
    summary: str = Field(..., description="Brief summary")
    key_points: List[str] = Field(..., description="Main points from the article")
    source: str = Field(..., description="News source")
    category: str = Field(..., description="Article category")


class StructuredOutputTutorial:
    """
    Demonstrates extracting structured data from the model.

    By giving the model JSON schema and examples, we can get reliable
    structured outputs that we can immediately use as Python objects.
    """

    def __init__(self):
        """Initialize the LangChain Bedrock model."""
        self.model = ChatBedrock(
            model_id=MODEL_ID,
            region_name="us-east-1",
            model_kwargs={"temperature": 0.7}
        )

    def _get_json_schema(self, model_class) -> dict:
        """
        Get the JSON schema for a Pydantic model.

        Args:
            model_class: A Pydantic model class

        Returns:
            JSON schema dict that describes the model
        """
        # Pydantic can generate JSON schema for validation
        return model_class.model_json_schema()

    def example_1_simple_extraction(self):
        """Example 1: Extract person info as structured data."""
        pretty_print("Example 1: Extract Person Info", "")

        # Get the JSON schema for our Person model
        schema = self._get_json_schema(Person)

        # Create a prompt that asks for structured output
        prompt = f"""Extract the person information from this text and respond with ONLY valid JSON matching this schema:

Schema: {json.dumps(schema, indent=2)}

Text: "Alice Johnson is 34 years old and works as a software engineer."

Return ONLY the JSON object, no other text."""

        print("Requesting structured person data...")
        print(f"Expected schema:\n{json.dumps(schema, indent=2)}\n")

        # Get response from model
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        response_text = response.content

        print(f"Model response:\n{response_text}\n")

        # Parse the JSON
        try:
            parsed_json = parse_json_from_response(response_text)
            # Validate with Pydantic
            person = Person(**parsed_json)
            print(f"[OK] Parsed and validated:")
            print(f"  Name: {person.name}")
            print(f"  Age: {person.age}")
            print(f"  Occupation: {person.occupation}\n")
        except (ValueError, ValidationError) as e:
            print(f"[FAIL] Failed to parse: {e}\n")

    def example_2_recipe_extraction(self):
        """Example 2: Extract recipe as structured data."""
        pretty_print("Example 2: Extract Recipe", "")

        schema = self._get_json_schema(Recipe)

        # Request a recipe in structured format
        prompt = f"""Create a simple recipe and respond with ONLY valid JSON matching this schema:

Schema: {json.dumps(schema, indent=2)}

Create a recipe for: Simple Pasta Carbonara

Return ONLY the JSON object, no other text."""

        print("Requesting recipe...")

        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        response_text = response.content

        print(f"Model response:\n{response_text}\n")

        # Parse and validate
        try:
            parsed_json = parse_json_from_response(response_text)
            recipe = Recipe(**parsed_json)
            print(f"[OK] Parsed recipe: {recipe.name}")
            print(f"  Cuisine: {recipe.cuisine}")
            print(f"  Servings: {recipe.servings}")
            print(f"  Time: {recipe.cooking_time_minutes} minutes")
            print(f"  Ingredients: {len(recipe.ingredients)} items")
            print(f"  Steps: {len(recipe.instructions)} steps\n")
        except (ValueError, ValidationError) as e:
            print(f"[FAIL] Failed to parse: {e}\n")

    def example_3_code_review(self):
        """Example 3: Code review with structured output."""
        pretty_print("Example 3: Code Review Analysis", "")

        schema = self._get_json_schema(CodeReview)

        # Sample code
        code = """
def add(a, b):
    return a + b
result = add(5, 3)
print(result)
"""

        prompt = f"""Review this code and respond with ONLY valid JSON matching this schema:

Schema: {json.dumps(schema, indent=2)}

Code to review:
{code}

Provide a thorough code review with issues, improvements, and a rating.

Return ONLY the JSON object, no other text."""

        print("Requesting code review...")

        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        response_text = response.content

        print(f"Model response:\n{response_text}\n")

        try:
            parsed_json = parse_json_from_response(response_text)
            review = CodeReview(**parsed_json)
            print(f"[OK] Code Review: {review.file_name}")
            print(f"  Rating: {review.rating}/10")
            print(f"  Issues found: {len(review.issues)}")
            for issue in review.issues:
                print(f"    - {issue}")
            print(f"  Improvements: {len(review.improvements)}")
            for imp in review.improvements:
                print(f"    - {imp}")
            print(f"  Overall: {review.overall_feedback}\n")
        except (ValueError, ValidationError) as e:
            print(f"[FAIL] Failed to parse: {e}\n")

    def example_4_multiple_items(self):
        """Example 4: Extract multiple items as structured data."""
        pretty_print("Example 4: Multiple Structured Items", "")

        # Create a model that holds multiple articles
        class NewsCollection(BaseModel):
            articles: List[NewsArticle] = Field(..., description="List of news articles")

        schema = self._get_json_schema(NewsCollection)

        prompt = f"""Generate 3 sample news articles and respond with ONLY valid JSON matching this schema:

Schema: {json.dumps(schema, indent=2)}

Create realistic but fictional news articles about technology.

Return ONLY the JSON object, no other text."""

        print("Requesting multiple news articles...")

        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)
        response_text = response.content

        print(f"Model response (first 300 chars):\n{response_text[:300]}...\n")

        try:
            parsed_json = parse_json_from_response(response_text)
            collection = NewsCollection(**parsed_json)
            print(f"[OK] Parsed {len(collection.articles)} articles:")
            for i, article in enumerate(collection.articles):
                print(f"  {i+1}. {article.headline} ({article.source})")
                print(f"     Category: {article.category}")
                print(f"     Points: {len(article.key_points)}\n")
        except (ValueError, ValidationError) as e:
            print(f"[FAIL] Failed to parse: {e}\n")

    def example_5_validation_demo(self):
        """Example 5: Show validation in action."""
        pretty_print("Example 5: Pydantic Validation", "")

        # Show valid data
        print("Valid data:")
        try:
            valid_person = Person(name="Bob Smith", age=28, occupation="Designer")
            print(f"[OK] Created: {valid_person.name}, {valid_person.age}, {valid_person.occupation}\n")
        except ValidationError as e:
            print(f"[FAIL] Error: {e}\n")

        # Show invalid data (wrong type)
        print("Invalid data (wrong type):")
        try:
            invalid_person = Person(name="Charlie", age="not_a_number", occupation="Manager")
            print(f"[OK] Created: {invalid_person}")
        except ValidationError as e:
            print(f"[FAIL] Validation failed: {e}\n")

        # Show another invalid example
        print("Invalid data (invalid code rating):")
        try:
            invalid_review = CodeReview(
                file_name="test.py",
                issues=["Issue 1"],
                improvements=["Improve 1"],
                rating=15,  # Out of 1-10 range
                overall_feedback="Good code"
            )
            print(f"[OK] Created: {invalid_review}")
        except ValidationError as e:
            print(f"[FAIL] Validation failed: {e}\n")

        print("[OK] Pydantic ensures all data matches the expected schema!\n")

    def demonstrate(self):
        """Run all examples."""
        print("\n" + "=" * 80)
        print("  LEVEL 4: Structured Output with Pydantic")
        print("=" * 80)

        try:
            self.example_1_simple_extraction()
            self.example_2_recipe_extraction()
            self.example_3_code_review()
            self.example_4_multiple_items()
            self.example_5_validation_demo()

            # Summary
            pretty_print("Summary", "")
            print("Key takeaways:")
            print("  1. Pydantic models define expected data structure")
            print("  2. JSON schema ensures model outputs match expectations")
            print("  3. Structured prompts guide models to output valid JSON")
            print("  4. Automatic validation ensures data integrity")
            print("  5. Type-safe Python objects from model outputs")
            print("  6. Can extract single items or lists of items")
            print("  7. Validation errors catch mismatches early")

        except Exception as e:
            pretty_print("ERROR", str(e))
            raise


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Entry point - demonstrates structured output with Pydantic.

    Usage:
        python level_04_structured_output.py
    """

    tutorial = StructuredOutputTutorial()
    tutorial.demonstrate()
