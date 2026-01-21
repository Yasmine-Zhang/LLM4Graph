from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from time import sleep


class BaseClient(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.config.get("name", "unknown_model")
        # Generation params
        self.top_p = self.config.get("top_p", 1.0)
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_new_tokens", 16)
        self.seed = self.config.get("seed", 42)
        
        # Retry logic
        self.max_attempts = self.config.get("max_attempts", 3)
        self.sleep_time = self.config.get("sleep_time", 1.0)

    def predict(self, message: str, candidates: List[str], system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """
        Predicts the class for the message text.
        Retries if the output is not in candidates or API fails.
        """
        sys_prompt = system_prompt if system_prompt is not None else ""
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": message}
        ]
        
        for attempt in range(self.max_attempts):
            try:
                response, log_prob = self.predict_once(messages=messages)
                
                # Check if response is valid
                response = response.strip()
                if response not in candidates:
                    raise ValueError(f"Output '{response}' not in candidates.")
                
                return response, log_prob
                
            except Exception as e:
                sleep(self.sleep_time)
                continue

        return candidates[0], -999.0

    @abstractmethod
    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        Abstract method to be implemented by specific clients.
        Should return (response_string, log_probability).
        """
        pass

def get_client(config: Dict) -> BaseClient:
    """
    Factory method to get the correct LLM client class.
    """
    client_type = config.get("type", "MockClient")
    system_prompt = config.get("system_prompt", "")
    
    if client_type == "MockClient":
        from .mock_client import MockClient
        return MockClient(config, system_prompt)
    elif client_type == "TransformersClient":
        # Placeholder for future implementation
        # from .transformers_client import TransformersClient
        # return TransformersClient(config, system_prompt)
        raise NotImplementedError("TransformersClient not yet implemented.")
    else:
        raise ValueError(f"Unknown client type: {client_type}")
