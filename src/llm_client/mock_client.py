from typing import List, Dict, Tuple
import random
import time
from .base_client import BaseClient

class MockClient(BaseClient):
    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        Simulate an LLM prediction.
        """
        # Simulate network latency
        time.sleep(self.sleep_time)

        candidates = [
            "cs.NA", "cs.MM", "cs.LO", "cs.CY", "cs.CR", "cs.DC", "cs.HC", "cs.CE",
            "cs.NI", "cs.CC", "cs.AI", "cs.MA", "cs.GL", "cs.NE", "cs.SC", "cs.AR",
            "cs.CV", "cs.GR", "cs.ET", "cs.SY", "cs.CG", "cs.OH", "cs.PL", "cs.SE",
            "cs.LG", "cs.SD", "cs.SI", "cs.RO", "cs.IT", "cs.PF", "cs.CL", "cs.IR",
            "cs.MS", "cs.FL", "cs.DS", "cs.OS", "cs.GT", "cs.DB", "cs.DL", "cs.DM"
        ]
        # Simple Random Pick
        category = random.choice(candidates)
        
        # Simple Random LogProb (High confidence range: -0.01 to -0.5)
        log_prob = random.uniform(-0.01, -0.5)

        return category, log_prob
