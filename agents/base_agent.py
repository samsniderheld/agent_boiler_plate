import yaml
from typing import Dict, Any, Optional, Union
from .llm_wrapper import LLMWrapper

class BaseAgent:
    """
    Base class for all agents.

    Attributes:
        config (Dict[str, Any]): Configuration loaded from the config file.
        llm (LLMWrapper): Wrapper for the language model.
        context (str): Context for the agent.
    """

    def __init__(self, config_file: str = None, llm: str = "openai", schema_path: Optional[str] = None) -> None:
        """
        Initializes the BaseAgent with a configuration file.

        Args:
            config_file (str): Path to the configuration file. Defaults to None.
            llm (str): LLM provider to use. Defaults to "openai".
            schema_path (Optional[str]): Path to YAML schema file for structured responses.
        """
        if config_file:
            self.config = self.load_config_file(config_file)
        else:
            self.config = self.default_config()
            
        self.llm = LLMWrapper(llm, schema_path=schema_path)
        self.name = self.config["name"]

    def load_config_file(self, config_file: str) -> Dict[str, Any]:
        """
        Loads the configuration file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def default_config(self) -> Dict[str, Any]:
        """
        Provides a default configuration.

        Returns:
            Dict[str, Any]: Default configuration dictionary.
        """
        return {
            "system_prompt": "Default system prompt.",
            "llm": "openAI"
        }

    def basic_api_call(self, query: str) -> str:
        """
        Makes a basic API call to the language model with the provided query.

        Args:
            query (str): The query to send to the language model.

        Returns:
            str: The response from the language model.
        """
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {"role": "user", "content": query}
        ]
        response = self.llm.make_api_call(messages)
        return response

    def basic_api_call_structured(self, query: str) -> Any:
        """
        Makes a basic API call to the language model with the provided query and expects a structured response.
        Uses the schema model set during agent initialization.

        Args:
            query (str): The query to send to the language model.

        Returns:
            Any: The structured response from the language model.
        """
        messages = [
            {
                "role": "system",
                "content": self.config['system_prompt']
            },
            {"role": "user", "content": query}
        ]
        response = self.llm.make_api_call_structured(messages)
        return response