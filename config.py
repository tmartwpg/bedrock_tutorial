"""
Shared configuration for the Bedrock tutorial.

This module centralizes all configuration settings used across levels,
making it easy to change parameters like model IDs, region, and defaults
without modifying individual level files.
"""

# ============================================================================
# AWS Configuration
# ============================================================================

# AWS region where Bedrock is available
# AWS_REGION = "us-east-1"
AWS_REGION = "ca-central-1"

# Model IDs available in Bedrock
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0" # Quicker, cheaper, slightly lower quality

# Alternative models you can try:
# MODEL_ID = "anthropic.claude-instant-v1:2:100k" # Not available in Canada
# MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
# MODEL_ID = "anthropic.claude-3-opus-20240229-v1:0" # More expensive, slower, better

# ============================================================================
# Model Pricing (per 1 million tokens)
# ============================================================================

MODEL_PRICING = {
    "anthropic.claude-opus-4-5-20251101-v1:0": (15.00, 75.00),
    "anthropic.claude-sonnet-4-5-20250929-v1:0": (3.00, 15.00),
    "anthropic.claude-haiku-4-5-20251001-v1:0": (0.80, 4.00),
    "anthropic.claude-3-sonnet-20240229-v1:0": (3.00, 15.00),
    "anthropic.claude-3-haiku-20240307-v1:0": (0.25, 1.25),
    "anthropic.claude-3-opus-20240229-v1:0": (15.00, 75.00),
}

# ============================================================================
# Model Parameters (Defaults)
# ============================================================================

# Temperature: Lower = more deterministic, Higher = more creative
# Range: 0.0 to 1.0
DEFAULT_TEMPERATURE = 0.7

# Maximum number of tokens in the response
# Why set this:
#   - Cost control: You pay per output token, limiting this reduces costs
#   - Response speed: Fewer tokens = faster inference and lower latency
#   - Prevent runaway: Without a limit, the model could generate very long responses
#   - Use-case specific: A summary needs ~100 tokens, detailed explanation needs ~2000
#   - Context window: Limited total context (input + output), so reserve space wisely
# Note: 1 token ≈ 4 characters or 0.75 words. Set higher for detailed responses (2048),
#       lower for concise ones (256). 1024 is a good middle ground.
DEFAULT_MAX_TOKENS = 1024

# Top-p (nucleus sampling): Controls diversity via cumulative probability
# Range: 0.0 to 1.0
# Why set this:
#   - How it works: Sorts tokens by probability, keeps the top ones until cumulative
#     probability reaches p (e.g., 0.9 = keep top ~10% most-likely tokens)
#   - Effect: Higher (0.95-1.0) = more random/creative; Lower (0.5-0.7) = more focused
#   - Use case: 0.9 is balanced for both creative and coherent responses
#   - Note: Works with temperature; they both increase randomness
# Typical values: 0.5 (focused), 0.9 (balanced - default), 1.0 (maximum diversity)
DEFAULT_TOP_P = 0.9

# Top-k: Only consider top k tokens by probability
# Range: 0 to 500
# Why set this:
#   - How it works: At each step, only sample from the k most-likely next tokens
#   - Effect: Prevents model from picking extremely unlikely tokens
#   - Use case: 250 excludes rare/nonsensical tokens while allowing creativity
#   - Note: Works alongside top-p; both filters apply
#   - Example: If top-k=250 but top-p=0.9 only needs 50 tokens, it uses 50
# Typical values: 0 (disabled), 50 (very restrictive), 250 (balanced - default)
DEFAULT_TOP_K = 250


# # News/facts - be precise
# temperature=0.3, top_k=50, top_p=0.5
# # Only samples from top-3 most likely tokens

# # Creative writing - allow variety
# temperature=0.8, top_k=100, top_p=0.9
# # Allows more options but still sensible

# # Code generation - balanced
# temperature=0.7, top_k=250, top_p=0.9
# # Good middle ground (Current)



# ============================================================================
# Timeout Configuration
# ============================================================================

# Request timeout in seconds
REQUEST_TIMEOUT = 60

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# ============================================================================
# LangChain Configuration
# ============================================================================

# Chunk size for document splitting (tokens)
CHUNK_SIZE = 500

# Overlap between chunks (tokens)
CHUNK_OVERLAP = 100

# Vector store configuration
VECTOR_STORE_SIMILARITY_TOP_K = 3

# ============================================================================
# Logging Configuration
# ============================================================================

# Set to True for verbose output
VERBOSE_LOGGING = True

# ============================================================================
# Example Texts & Prompts
# ============================================================================

# Sample text for RAG examples (Level 8)
SAMPLE_DOCUMENT = """
AWS Bedrock is a fully managed service that makes base models from leading AI companies
accessible through an API. You can choose from a wide range of foundation models to find
the model that best suits your use case.

Bedrock handles infrastructure management, so you don't have to manage servers or
containers. With Bedrock, you can:
- Access models from Anthropic, Cohere, Meta, Mistral, and others
- Use models with different capabilities: text generation, multimodal, and embeddings
- Take advantage of built-in features like knowledge bases and guardrails
- Scale automatically as demand changes

Bedrock's pricing is based on the number of input and output tokens processed by the model.
"""

# Sample tool definitions (for Level 5)
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
]

