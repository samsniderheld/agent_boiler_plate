import os
import json
import yaml
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Union
from datetime import datetime

from google import genai
from google.genai import types
from openai import OpenAI
from pydantic import BaseModel, Field, create_model


class SchemaGenerator:
    """
    Generates Pydantic models from YAML schema definitions.
    """
    
    @staticmethod
    def from_yaml_file(yaml_file: str) -> Type[BaseModel]:
        """
        Create a Pydantic model from a YAML schema file.
        
        Args:
            yaml_file: Path to the YAML schema file
            
        Returns:
            Pydantic model class
        """
        with open(yaml_file, 'r') as file:
            schema = yaml.safe_load(file)
        return SchemaGenerator.from_dict(schema)
    
    @staticmethod
    def from_dict(schema: Dict[str, Any]) -> Type[BaseModel]:
        """
        Create a Pydantic model from a schema dictionary.
        
        Args:
            schema: Schema dictionary with model name and fields
            
        Returns:
            Pydantic model class
        """
        model_name = schema.get('model_name', 'GeneratedModel')
        fields = schema.get('fields', {})
        
        # Convert YAML field definitions to Pydantic field definitions
        pydantic_fields = {}
        
        for field_name, field_def in fields.items():
            if isinstance(field_def, str):
                # Simple type definition
                field_type = SchemaGenerator._get_python_type(field_def)
                pydantic_fields[field_name] = (field_type, ...)
            elif isinstance(field_def, dict):
                # Complex field definition with type, description, default, etc.
                field_type = SchemaGenerator._get_python_type(field_def.get('type', 'str'), field_def.get('schema'))
                field_description = field_def.get('description', '')
                field_default = field_def.get('default', ...)
                
                if field_default == ...:
                    pydantic_fields[field_name] = (field_type, Field(description=field_description))
                else:
                    pydantic_fields[field_name] = (field_type, Field(default=field_default, description=field_description))
        
        # Create the model class
        model_class = create_model(model_name, **pydantic_fields)
        
        # Add custom methods if specified in schema
        if 'methods' in schema:
            SchemaGenerator._add_custom_methods(model_class, schema['methods'])
        
        return model_class
    
    @staticmethod
    def _get_python_type(type_str: str, nested_schema: Optional[Dict] = None) -> Type:
        """Convert YAML type string to Python type."""
        type_mapping = {
            'str': str,
            'string': str,
            'int': int,
            'integer': int,
            'float': float,
            'bool': bool,
            'boolean': bool,
            'list': list,
            'dict': dict,
            'any': Any,
        }
        
        # Handle list types like "list[dict]" with nested schema
        if type_str.startswith('list[') and type_str.endswith(']'):
            inner_type_str = type_str[5:-1]
            if inner_type_str == 'dict' and nested_schema:
                # Create nested model for the dict structure
                nested_model = SchemaGenerator._create_nested_model(nested_schema)
                return List[nested_model]
            else:
                inner_type = SchemaGenerator._get_python_type(inner_type_str)
                return List[inner_type]
        
        # Handle dict types with nested schema
        if type_str == 'dict' and nested_schema:
            return SchemaGenerator._create_nested_model(nested_schema)
        elif type_str.startswith('dict[') and type_str.endswith(']'):
            return Dict[str, Any]  # Simplified for now
        
        return type_mapping.get(type_str.lower(), str)
    
    @staticmethod
    def _create_nested_model(schema_dict: Dict[str, Any]) -> Type[BaseModel]:
        """Create a nested Pydantic model from a schema dictionary."""
        nested_fields = {}
        
        for field_name, field_def in schema_dict.items():
            if isinstance(field_def, str):
                field_type = SchemaGenerator._get_python_type(field_def)
                nested_fields[field_name] = (field_type, ...)
            elif isinstance(field_def, dict):
                field_type = SchemaGenerator._get_python_type(field_def.get('type', 'str'), field_def.get('schema'))
                field_description = field_def.get('description', '')
                field_default = field_def.get('default', ...)
                
                if field_default == ...:
                    nested_fields[field_name] = (field_type, Field(description=field_description))
                else:
                    nested_fields[field_name] = (field_type, Field(default=field_default, description=field_description))
        
        return create_model('NestedModel', **nested_fields)
    
    @staticmethod
    def _add_custom_methods(model_class: Type[BaseModel], methods: Dict[str, str]):
        """Add custom methods to the generated model class."""
        for method_name, method_body in methods.items():
            if method_name == 'to_json':
                def to_json(self):
                    return self.model_dump()
                setattr(model_class, 'to_json', to_json)
            elif method_name == 'to_str':
                def to_str(self):
                    return str(self.model_dump())
                setattr(model_class, 'to_str', to_str)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the interface that all LLM providers must implement.
    """
    
    @abstractmethod
    def make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        """
        Make a standard API call to the LLM.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response text from the LLM
        """
        pass
    
    @abstractmethod
    def make_api_call_structured(
        self, 
        messages: List[Dict[str, Any]], 
        response_model: Optional[Union[Type[BaseModel], str]] = None
    ) -> Any:
        """
        Make a structured API call to the LLM.
        
        Args:
            messages: List of message dictionaries
            response_model: Pydantic model class or path to YAML schema file
            
        Returns:
            Parsed response object
        """
        pass


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
    
    def make_api_call_structured(
        self, 
        messages: List[Dict[str, Any]], 
        response_model: Optional[Union[Type[BaseModel], str]] = None
    ) -> Any:
        # Handle YAML schema file
        if isinstance(response_model, str):
            response_model = SchemaGenerator.from_yaml_file(response_model)
        
        if response_model is None:
            # Default fallback model
            response_model = create_model('DefaultResponse', content=(str, Field(description="Response content")))
        
        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=response_model
        )
        return response.choices[0].message.parsed


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
    
    def make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=messages[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=messages[0]["content"]
            )
        )
        return response.text
    
    def make_api_call_structured(
        self, 
        messages: List[Dict[str, Any]], 
        response_model: Optional[Union[Type[BaseModel], str]] = None
    ) -> Any:
        # Handle YAML schema file
        if isinstance(response_model, str):
            response_model = SchemaGenerator.from_yaml_file(response_model)
        
        if response_model is None:
            # Default fallback model
            response_model = create_model('DefaultResponse', content=(str, Field(description="Response content")))
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=messages[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=messages[0]["content"],
                response_mime_type='application/json',
                response_schema=response_model
            )
        )
        
        # Parse JSON response into the model
        try:
            response_data = json.loads(response.text)
            return response_model(**response_data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            # Fallback: try to create model with the raw text
            try:
                return response_model(content=response.text)
            except:
                raise ValueError(f"Could not parse structured response: {e}")


class LLMProviderFactory:
    """
    Factory class for creating LLM providers.
    """
    
    _providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> BaseLLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of the provider ("openai" or "gemini")
            **kwargs: Additional arguments to pass to the provider constructor
            
        Returns:
            LLM provider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name.lower() not in cls._providers:
            raise ValueError(f"Unsupported provider: {provider_name}. Supported providers: {list(cls._providers.keys())}")
        
        provider_class = cls._providers[provider_name.lower()]
        return provider_class(**kwargs)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a new LLM provider.
        
        Args:
            name: Name of the provider
            provider_class: Provider class that inherits from BaseLLMProvider
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError("Provider class must inherit from BaseLLMProvider")
        cls._providers[name.lower()] = provider_class


class LLMWrapper:
    """
    Wrapper class for different language models.
    
    This class provides a unified interface using the provider pattern.
    """
    
    def __init__(self, provider: str = "gemini", schema_path: Optional[str] = None, **provider_kwargs) -> None:
        """
        Initializes the LLMWrapper with the specified provider.
        
        Args:
            provider (str): The provider to use ("openai" or "gemini")
            schema_path (Optional[str]): Path to YAML schema file for structured responses
            **provider_kwargs: Additional arguments to pass to the provider
        """
        self.provider_name = provider
        self.provider = LLMProviderFactory.create_provider(provider, **provider_kwargs)
        
        # Pre-generate schema model if provided
        self.default_schema_model = None
        if schema_path and os.path.exists(schema_path):
            try:
                self.default_schema_model = SchemaGenerator.from_yaml_file(schema_path)
            except Exception as e:
                print(f"Warning: Could not load schema from {schema_path}: {e}")

    def make_api_call(self, messages: List[Dict[str, Any]]) -> str:
        """
        Makes an API call to the language model with the provided messages.
        
        Args:
            messages: The messages to send to the language model.
            
        Returns:
            The response from the language model.
        """
        return self.provider.make_api_call(messages)
    
    def make_api_call_structured(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Makes a structured API call to the language model with the provided messages.
        Uses the default schema model set during initialization.
        
        Args:
            messages: The messages to send to the language model.
            
        Returns:
            The structured response from the language model.
        """
        return self.provider.make_api_call_structured(messages, self.default_schema_model)
