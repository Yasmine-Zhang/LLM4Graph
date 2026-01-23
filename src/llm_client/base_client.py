import re
import os
import json
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from time import sleep
from src.util.util import parse_node_text

class BaseClient(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.name = self.config.get("name", "unknown_model")
        # Generation params
        self.top_p = self.config.get("top_p", 1.0)
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_new_tokens", 16)
        self.seed = self.config.get("seed", 42)
        self.system_prompt = self.config.get("system_prompt", "")
        
        # Retry logic
        self.max_attempts = self.config.get("max_attempts", 3)
        self.sleep_time = self.config.get("sleep_time", 1.0)
        self.save_interval = self.config.get("save_interval", 100)

    def run_inference(self, loader, target_indices, output_dir, logger, prompt_template, candidates, llm_cache=None):
        """
        Run LLM inference or load from cache.
        Returns a dictionary of predictions: {node_idx: {'llm_predict': str, 'llm_confident': float}}
        """
        logger.info(f"Starting LLM Prediction Phase with {self.name}...")
        
        # Priority 1: User specified cache
        # Priority 2: Default location in output_dir
        llm_cache_path = llm_cache if llm_cache else os.path.join(output_dir, "llm_predict.json")
        
        predictions = {}
        
        # Load existing predictions if available (Resume capability)
        if os.path.exists(llm_cache_path):
            logger.info(f"Loading existing predictions from: {llm_cache_path}")
            try:
                with open(llm_cache_path, 'r') as f:
                    loaded_preds = json.load(f)
                    
                # Normalize keys to integers
                for k, v in loaded_preds.items():
                    node_idx = int(k)
                    # Normalize value format
                    cat = v.get('llm_predict', v.get('category'))
                    conf = v.get('llm_confident', v.get('confidence'))
                    raw = v.get('llm_raw_response', "")
                    predictions[node_idx] = {
                        "llm_predict": cat,
                        "llm_confident": conf,
                        "llm_raw_response": raw
                    }
                logger.info(f"Loaded {len(predictions)} predictions. Resuming...")
            except json.JSONDecodeError:
                logger.warning(f"Cache file {llm_cache_path} corrupted or empty. Starting from scratch.")
                
        # Determine which nodes still need prediction
        remaining_indices = [idx for idx in target_indices if idx not in predictions]
        
        if not remaining_indices:
            logger.info("All target nodes have been predicted. Skipping inference.")
        else:
            logger.info(f"Remaining nodes to predict: {len(remaining_indices)}")
            
            save_path = os.path.join(output_dir, "llm_predict.json")
            processed_count = 0
            
            # Process remaining nodes
            for idx in tqdm(remaining_indices, desc="LLM Inference"):
                raw_node_text = loader.get_formatted_message(idx)
                title, abstract = parse_node_text(raw_node_text)
                
                # Format prompt
                message = prompt_template.format(title=title, abstract=abstract)
                
                # Predict (Using self.system_prompt)
                pred_cat, log_prob, raw_response = self.predict(message=message, candidates=candidates, system_prompt=self.system_prompt)
                
                # Update strictly locally
                predictions[idx] = {
                    "llm_predict": pred_cat,
                    "llm_confident": log_prob,
                    "llm_raw_response": raw_response 
                }
                
                processed_count += 1
                
                # Periodic Save
                if processed_count % self.save_interval == 0:
                    with open(save_path, 'w') as f:
                        json.dump(predictions, f, indent=2)
                    logger.info(f"Progress Checkpoint: Processed {processed_count}/{len(remaining_indices)} nodes. Saved predictions into {save_path}")

            # Final Save after loop completes
            with open(save_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            logger.info(f"Inference completed. Total predictions saved to {save_path}")
            
        return predictions

    def predict(self, message: str, candidates: List[str], system_prompt: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Predicts the class for the message text.
        Retries if the output is not in candidates or API fails.
        Returns: (predicted_category, confidence_score, full_response_text)
        """
        sys_prompt = system_prompt if system_prompt is not None else ""
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": message}
        ]
        
        last_response = ""
        
        for attempt in range(self.max_attempts):
            try:
                response, log_prob = self.predict_once(messages=messages)
                last_response = response # Keep track for fallback
                
                # CoT Parsing Logic
                # Expected format: "Analysis: ... Category: cs.XX"
                # Fallback: Just "cs.XX"
                
                clean_response = response.strip()
                extracted_cat = clean_response
                
                # Regex to find "Category: cs.XX" or just "cs.XX" at the end
                # Pattern: Look for 'cs.[A-Z]{2}' case insensitive
                # Prioritize 'Category:' marker if present
                
                if "Category:" in clean_response:
                    parts = clean_response.split("Category:")
                    candidate_part = parts[-1].strip()
                    extracted_cat = candidate_part.split()[0] if candidate_part else ""
                # Handle JSON block ```json ... ```
                elif "```json" in clean_response:
                    try:
                        import json
                        # finding the json block
                        start = clean_response.find("{")
                        end = clean_response.rfind("}") + 1
                        json_str = clean_response[start:end]
                        data = json.loads(json_str)
                        # Try to extract category code? Wait, user loader prompt outputs ID not Code.
                        # This implies a prompt mismatch between loader and config.
                        # If we get here, it means the model is following the loader's default prompt
                        # instead of the config's prompt. 
                        # We should fix main.py to prioritize config prompt.
                        pass
                    except:
                        pass
                
                if not extracted_cat:
                    extracted_cat = clean_response
                    # Fallback Strategy:
                    # If model didn't use "Category:", maybe it just outputted the code at the end
                    # Or maybe it outputted "cs.AI" in the middle.
                    pass 

                # Clean up formatting artifacts (punctuation, brackets)
                if extracted_cat:
                    extracted_cat = extracted_cat.split(" (")[0].strip()
                    extracted_cat = extracted_cat.rstrip(".").strip()
                    if "\n" in extracted_cat:
                        extracted_cat = extracted_cat.split("\n")[0].strip()

                # Normalize candidates to handle case sensitivity
                response_lower = extracted_cat.lower()
                candidates_lower = [c.lower() for c in candidates]
                
                if response_lower not in candidates_lower:
                    err_msg = f"Output '{extracted_cat}' (from '{clean_response}') not in candidates."
                    # On final attempt, print error to console
                    if attempt == self.max_attempts - 1:
                        print(f"[ERROR Final] {err_msg}")
                    raise ValueError(err_msg)
                
                # Restore the correct candidate string format
                match_idx = candidates_lower.index(response_lower)
                canonical_response = candidates[match_idx]
                
                return canonical_response, log_prob, response
                
            except Exception as e:
                sleep(self.sleep_time)
                continue

        # If we failed all attempts, return the last raw response so we can debug it
        return candidates[0], -999.0, last_response

    @abstractmethod
    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, float]:
        """
        Abstract method to be implemented by clients.
        Should return (full_text_response, confidence_score).
        """
        pass


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
