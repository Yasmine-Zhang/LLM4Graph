from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

class BaseLLMClient(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.config.get("name", "unknown_model")
        self.top_p = self.config.get("top_p", self.config.get("top-p", 1.0))
        self.temperature = self.config.get("temperature", 1.0)
        self.max_tokens = self.config.get("max_tokens", 32000)
        self.seed = self.config.get("seed", None)
        self.think = self.config.get("think", False)
    
    @abstractmethod
    def predict(self, message: str, system_prompt: Optional[str] = None) -> List[Tuple[str, float]]:
        pass
