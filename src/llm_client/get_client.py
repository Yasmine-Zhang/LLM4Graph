from typing import Dict
from src.llm_client.base_client import BaseClient


def get_client(config: Dict) -> BaseClient:
    # Strictly enforce Dict config (from YAML)
    if not isinstance(config, dict):
        raise ValueError(f"Expected config dict, got: {type(config)}")

    llm_type = config.get("type", "MockClient") # Default to Transformers if missing
    
    if llm_type == "MockClient":
        from src.llm_client.mock_client import MockClient
        llm_client = MockClient(config=config)
    else:
        raise ValueError(f"Unknown client type: {llm_type}")
    return llm_client