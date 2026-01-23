from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseDataLoader(ABC):
    """
    Abstract Base Class for Graph Datasets (LLM+Graph).
    Ensures consistent interface for data loading, text retrieval, and prompt generation.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.root = config.get('root', 'dataset')
        self.data = None # Should be populated by implementation
        self.split_idx = None # Should be populated by implementation

    @abstractmethod
    def get_data(self) -> Any:
        """
        Return the PyG data object (Data).
        """
        pass
    
    @abstractmethod
    def get_formatted_message(self, node_idx: int) -> str:
        """
        Return the formatted user message (e.g. Title + Abstract) for a specific node.
        This is what will be fed to the LLM as the user query.
        """
        pass
        
    def get_system_prompt(self) -> str:
        """
        Return the system prompt used for LLM inference.
        Priority:
            1. Config-defined 'system_prompt' (if present and not empty)
            2. Dataset-specific default prompt (_get_default_system_prompt)
        """
        if 'system_prompt' in self.config and self.config['system_prompt']:
            return self.config['system_prompt']
        else:
            return "You are a helpful assistant for graph node classification."

    def _get_default_system_prompt(self) -> str:
        """
        Returns a default system prompt specific to the dataset.
        Subclasses should override this if they have a specific default behavior.
        """

    @abstractmethod
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Return mapping from label index to raw label string (e.g., {0: 'arxiv cs.AI'}).
        """
        pass
        
    @abstractmethod
    def get_inv_label_map(self) -> Dict[str, int]:
        """
        Return standardized mapping from normalized label string to index (e.g., {'CS.AI': 0}).
        Used for converting LLM string outputs back to class indices.
        """
        pass
