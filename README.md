# AWS Bedrock Tutorial: From Hello World to Advanced LLM Applications

A comprehensive, progressive tutorial that builds your understanding of working with AWS Bedrock and LangChain, starting from basic model invocation to advanced patterns like RAG, structured outputs, and error handling.

## Overview

This tutorial is structured as 10 independent, runnable levels. Each level builds on concepts from previous levels, but can be executed standalone. Code is designed to be:
- **Clean & Readable**: Well-commented at script, function, and row level
- **Progressive**: Incrementally more complex patterns
- **Practical**: Real-world examples and use cases
- **Reusable**: Shared utilities to avoid duplication

## Requirements

- Python 3.11+
- AWS credentials configured locally (with Bedrock access)
- AWS Bedrock model access (Claude Sonnet 3.5)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Levels

### Level 1: Hello World
**File**: `level_01_hello_world.py`

The foundation. Connect to AWS Bedrock and invoke Claude Sonnet to get a simple response.

```bash
python level_01_hello_world.py
```

**Topics**: AWS SDK setup, basic model invocation, response parsing

---

### Level 2: Basic Prompting
**File**: `level_02_basic_prompting.py`

Explore prompt structure, system prompts, user prompts, and model parameters (temperature, max tokens).

```bash
python level_02_basic_prompting.py
```

**Topics**: System prompts, temperature, max_tokens, model parameters

---

### Level 3: Prompt Templates
**File**: `level_03_prompt_templates.py`

Use LangChain prompt templates to create reusable prompt patterns with variable substitution.

```bash
python level_03_prompt_templates.py
```

**Topics**: LangChain templates, prompt reusability, variable injection

---

### Level 4: Structured Output with Pydantic
**File**: `level_04_structured_output.py`

Get the model to output structured JSON that you can parse into Pydantic models for type safety and validation.

```bash
python level_04_structured_output.py
```

**Topics**: Pydantic models, JSON schema, structured extraction, validation

---

### Level 5: Tool Use & Function Calling
**File**: `level_05_tool_use.py`

Enable the model to call functions/tools, process results, and continue reasoning (agentic behavior).

```bash
python level_05_tool_use.py
```

**Topics**: Tool definitions, function calling, agent loops, tool results

---

### Level 6: Conversation Memory
**File**: `level_06_conversation_memory.py`

Build multi-turn conversations where the model remembers previous context using conversation buffers.

```bash
python level_06_conversation_memory.py
```

**Topics**: Conversation history, memory management, multi-turn dialogs, context windows

---

### Level 7: Chains
**File**: `level_07_chains.py`

Combine multiple operations into sequences using LangChain chains for complex workflows.

```bash
python level_07_chains.py
```

**Topics**: LangChain chains, sequential operations, workflow composition

---

### Level 8: Vector Store & RAG (Retrieval-Augmented Generation)
**File**: `level_08_vector_store_rag.py`

Build a retrieval system where the model answers questions based on a document knowledge base.

```bash
python level_08_vector_store_rag.py
```

**Topics**: Embeddings, vector stores, document chunking, semantic search, RAG

---

### Level 9: Error Handling & Retries
**File**: `level_09_error_handling.py`

Build resilient applications with retry logic, error handling, rate limiting, and graceful degradation.

```bash
python level_09_error_handling.py
```

**Topics**: Exception handling, retry strategies, rate limiting, circuit breakers, fallbacks

---

### Level 10: Streaming & Advanced Patterns
**File**: `level_10_streaming_advanced.py`

Stream model responses for real-time output, async operations, and combine advanced patterns.

```bash
python level_10_streaming_advanced.py
```

**Topics**: Streaming responses, async/await, token counting, batch processing, performance optimization

---

## Project Structure

```
bedrock_tutorial/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Shared configuration
├── utils.py                     # Shared utilities (setup, helpers)
│
├── level_01_hello_world.py     # Basic connection & invocation
├── level_02_basic_prompting.py # Prompt structure
├── level_03_prompt_templates.py # LangChain templates
├── level_04_structured_output.py # Pydantic + JSON schema
├── level_05_tool_use.py        # Function calling & agents
├── level_06_conversation_memory.py # Multi-turn conversations
├── level_07_chains.py          # LangChain chains
├── level_08_vector_store_rag.py # RAG & embeddings
├── level_09_error_handling.py  # Resilience patterns
├── level_10_streaming_advanced.py # Streaming & async
│
└── main.py                      # Optional: Run all levels
```

## Shared Utilities

**config.py**: Centralized configuration
- Model IDs
- AWS region
- Common parameters

**utils.py**: Helper functions
- Bedrock client initialization
- Pretty printing
- Common setup steps

## Running Individual Levels

Each level is a complete, runnable Python script:

```bash
python level_01_hello_world.py
python level_02_basic_prompting.py
# ... etc
```

Each includes:
- Example usage (main/demo section)
- Expected output comments
- Explanatory comments throughout

## Dependencies

See `requirements.txt` for full list. Key packages:
- `boto3`: AWS SDK
- `langchain`: LLM orchestration
- `langchain-community`: Community integrations
- `pydantic`: Data validation
- `langchain-aws`: LangChain Bedrock integration
- `faiss-cpu`: Vector similarity search (for RAG)

## Tips for Learning

1. Start with Level 1 and work sequentially
2. Each level comments show what's happening
3. Modify examples and experiment
4. Combine patterns from multiple levels for real applications
5. Check AWS Bedrock console for token usage and costs
6. Refer to LangChain docs for deeper understanding

## Common Issues

**"No credentials found"**: Ensure AWS credentials are configured
```bash
aws configure
```

**"Model not found"**: Verify Bedrock model access in your AWS account

**"Rate limited"**: See Level 9 for retry strategies

## Next Steps

After completing all levels:
- Build a real application combining levels
- Add persistence (databases)
- Deploy with Lambda + API Gateway
- Explore other models available in Bedrock
- Experiment with fine-tuning

## Resources

- [AWS Bedrock Docs](https://docs.aws.amazon.com/bedrock/)
- [LangChain Documentation](https://python.langchain.com/)
- [Claude Model Cards](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [LangChain Bedrock Integration](https://python.langchain.com/docs/integrations/llms/bedrock/)

---

**Happy Learning!** Start with Level 1 and progressively build your LLM expertise.
