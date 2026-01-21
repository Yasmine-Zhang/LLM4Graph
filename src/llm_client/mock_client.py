from typing import List, Dict, Tuple
import random
from src.llm_client.base_client import BaseClient

class MockClient(BaseClient):
    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        Simulate an LLM prediction.
        """

        candidates = [
            "cs.NA", "cs.MM", "cs.LO", "cs.CY", "cs.CR", "cs.DC", "cs.HC", "cs.CE",
            "cs.NI", "cs.CC", "cs.AI", "cs.MA", "cs.GL", "cs.NE", "cs.SC", "cs.AR",
            "cs.CV", "cs.GR", "cs.ET", "cs.SY", "cs.CG", "cs.OH", "cs.PL", "cs.SE",
            "cs.LG", "cs.SD", "cs.SI", "cs.RO", "cs.IT", "cs.PF", "cs.CL", "cs.IR",
            "cs.MS", "cs.FL", "cs.DS", "cs.OS", "cs.GT", "cs.DB", "cs.DL", "cs.DM"
        ]
        # Simple Random Pick
        category = random.choice(candidates)
        
        # Simple Random LogProb (Simulate varied confidence)
        # 30% chance of high confidence (> 0.8)
        # 70% chance of low confidence (< 0.8)
        if random.random() < 0.3:
             # High confidence: -0.01 (0.99) to -0.22 (0.80)
             log_prob = random.uniform(-0.01, -0.22) 
        else:
             # Low confidence: -0.23 (0.79) to -5.0 (0.006)
             log_prob = random.uniform(-0.23, -5.0)

        return category, log_prob
