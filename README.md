# Agent Boilerplate

A flexible boilerplate for creating and managing AI agents with multiple LLM providers. Features an abstract provider system with dynamic schema generation from YAML files for structured responses. Supports OpenAI, Google Gemini, and custom LLM providers.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/samsniderheld/agent_boiler_plate.git
    cd agent_boiler_plate
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install extra dependencies:

    ```sh
    pip install torchao --extra-index-url https://download.pytorch.org/whl/cu121 # full options are cpu/cu118/cu121/cu124
    pip install git+https://github.com/xhinker/sd_embed.git@main
    ```

## API Configurations

The system supports multiple LLM providers. Configure the APIs you want to use:

### OpenAI
1. Obtain your OpenAI API key from the [OpenAI website](https://platform.openai.com/api-keys)
2. Set your API key as an environment variable:
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    ```

### Google Gemini
1. Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set your API key as an environment variable:
    ```sh
    export GEMINI_API_KEY='your_gemini_api_key'
    ```

## Agent Configuration

Agents are configured using YAML files in the `config_files/` directory.

Example `example.yaml`:

```yaml
name: example
system_prompt: |
  You are a creative agent. Your job is to take an input prompt and come up with some creative concepts for it.
```

## Architecture

### LLM Provider System
The system uses an abstract provider pattern for maximum flexibility:

- **`BaseLLMProvider`**: Abstract base class defining the interface
- **`OpenAIProvider`**: OpenAI GPT models implementation
- **`GeminiProvider`**: Google Gemini models implementation  
- **`LLMProviderFactory`**: Factory for creating and registering providers
- **`LLMWrapper`**: Unified interface with schema pre-loading
- **`SchemaGenerator`**: Dynamic Pydantic model generation from YAML

### BaseAgent
The `BaseAgent` class loads configuration files and provides both standard and structured API interfaces:

```python
# Create agent with specific LLM provider and schema
agent = BaseAgent(
    config_file="config_files/example.yaml", 
    llm="openai",
    schema_path="config_files/creative_response_schema.yaml"
)

# Standard API calls
response = agent.basic_api_call("Your prompt here")

# Structured API calls (uses pre-loaded schema)
structured = agent.basic_api_call_structured("Create creative concepts")
print(structured.title, structured.concept)
```

## Quick Start

### Interactive Example
Run the interactive example agent with both standard and structured response capabilities:

```sh
python3 example_agent.py
```

This creates an agent from `config_files/example.yaml` with pre-loaded schema and starts an interactive chat loop.

**Available commands:**
- Regular chat: `Create concepts for a coffee shop`
- Structured responses: `structured Create concepts for a sustainable coffee shop`
- View schema: `schema`
- Help: `help`

### Programmatic Usage

```python
from agents.base_agent import BaseAgent

# Create agent with Gemini and pre-loaded schema
agent = BaseAgent(
    config_file="config_files/example.yaml",
    llm="gemini",
    schema_path="config_files/creative_response_schema.yaml"
)

# Standard chat
response = agent.basic_api_call("Come up with creative concepts for a coffee shop")
print(response)

# Structured response (uses pre-loaded schema)
structured = agent.basic_api_call_structured("Create sustainable coffee shop concepts")
print(f"Title: {structured.title}")
print(f"Concept: {structured.concept}")
print(f"Target Audience: {structured.target_audience}")
print(f"Feasibility: {structured.feasibility_score}/10")
```

## Dynamic Schema Generation

Define response structures using YAML schemas that are automatically converted to Pydantic models:

### Schema Definition

`config_files/creative_response_schema.yaml`:
```yaml
model_name: CreativeResponse
description: Schema for creative concept responses
fields:
  title:
    type: str
    description: The title of the creative concept
  concept:
    type: str
    description: Detailed description of the creative concept
  key_features:
    type: list[str]
    description: List of key features or highlights
  target_audience:
    type: str
    description: Who this concept would appeal to
  feasibility_score:
    type: int
    description: Feasibility score from 1-10
    default: 5
methods:
  to_json: true
  to_str: true
```

### Supported Field Types
- `str`, `int`, `float`, `bool`
- `list[str]`, `list[int]`, etc.
- `dict[str, any]`
- Complex nested structures

### Schema Features
- **Field descriptions**: Help LLMs understand expected content
- **Default values**: Optional fields with fallback values
- **Custom methods**: Add `to_json()` and `to_str()` methods
- **Type validation**: Automatic Pydantic validation

## Extending with Custom Providers

Add support for new LLM providers:

```python
from agents.llm_wrapper import BaseLLMProvider, LLMProviderFactory

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet"):
        # Initialize Anthropic client
        pass
    
    def make_api_call(self, messages):
        # Implement API call logic
        pass
    
    def make_api_call_structured(self, messages, response_model):
        # Implement structured response logic  
        pass

# Register the new provider
LLMProviderFactory.register_provider("anthropic", AnthropicProvider)

# Use it with schema
agent = BaseAgent(
    config_file="config.yaml", 
    llm="anthropic",
    schema_path="config_files/my_schema.yaml"
)
```

## Project Structure

```
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # Base agent class
│   └── llm_wrapper.py             # LLM provider system & schema generator
├── config_files/
│   ├── example.yaml               # Example agent configuration
│   └── creative_response_schema.yaml # Example YAML schema
├── example_agent.py               # Interactive example script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md
```

## Features

- **Multi-Provider Support**: Easy switching between OpenAI, Gemini, and custom providers
- **Dynamic Schema Generation**: Create Pydantic models from YAML schema definitions
- **Structured Responses**: Get consistent, typed responses from LLMs
- **Schema Pre-loading**: Efficient architecture that loads schemas once during initialization
- **Abstract Provider System**: Clean architecture for adding new LLM providers
- **YAML Configuration**: Simple agent and schema configuration using YAML files
- **Interactive Interface**: Ready-to-use chat interface with structured response demo
- **Type Safety**: Full type hints for better development experience
- **Error Handling**: Comprehensive error handling with graceful fallbacks
- **Extensible**: Factory pattern allows easy registration of custom providers

## Performance Benefits

- **Efficient Schema Loading**: YAML schemas are parsed and converted to Pydantic models once during agent initialization, not on every API call
- **Memory Optimization**: Pre-loaded schema models eliminate repetitive parsing overhead
- **Fast Structured Responses**: Direct use of compiled Pydantic models for validation and serialization

## License

MIT License - see LICENSE file for details.
