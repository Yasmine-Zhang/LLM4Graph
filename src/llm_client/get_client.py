from typing import Dict
from src.llm_client.base_client import BaseClient


def get_client(config: Dict) -> BaseClient:
    # Strictly enforce Dict config (from YAML)
    if not isinstance(config, dict):
        raise ValueError(f"Expected config dict, got: {type(config)}")

    llm_type = config.get("type", "AzureGPT") # Default to AzureGPT if missing
    
    if llm_type == "AzureGPT":
        from src.llm_client.azure_gpt import AzureGPTClient
        llm_client = AzureGPTClient(config=config)
    else:
        raise ValueError(f"Unknown client type: {llm_type}")
    return llm_client