import os
import torch
import transformers
from typing import List, Dict, Optional, Tuple
from src.llm_client.base_client import BaseClient
import logging

class TransformersClient(BaseClient):
    """
    Client for local Transformers models (HuggingFace).
    Supports extraction of logprobs from local generation.
    """
    def __init__(
            self,
            config: Dict,
        ):
        super().__init__(config)
        
        # In new config.yaml, the field is 'name' (e.g., "Qwen/Qwen3-8B")
        model_name = self.config.get('name')
        
        # Thinking parameter
        self.think = self.config.get("think", False)

        if not model_name:
             raise ValueError(f"Config missing 'name' field for model. Config: {self.config}")
         
        if os.getenv("AMLT_DATA_DIR"):
            self.model_path = os.path.join(os.getenv("AMLT_DATA_DIR"), os.path.normpath(model_name))
        else:
            self.model_path = os.path.normpath(model_name)

        self.device_id = self.config.get("device_id", 0)
        device = self.device_id if torch.cuda.is_available() else -1

        # Attention Implementation Selection
        target_attn_impl = self.config.get("attn_implementation", "sdpa")
        # Ministral prefers eager or specific implementations, but SDPA is generally safe for modern Torch
        if "Ministral" in str(self.model_path) or "ministral" in str(self.model_path).lower():
            target_attn_impl = "eager"
            
        logger_local = logging.getLogger(__name__)

        # Initialize Pipeline (Minimal logic)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={
                "dtype": torch.bfloat16,
                "attn_implementation": target_attn_impl, 
            },
            device=device,
            trust_remote_code=True,
        )
        
        print(f"DEBUG: Model {self.model_path} loaded on device: {self.pipeline.model.device}. requested device_id: {self.device_id}")

        # Ensure pad_token is set
        if self.pipeline.tokenizer.pad_token_id is None:
            self.pipeline.tokenizer.pad_token_id = self.pipeline.tokenizer.eos_token_id
            if self.pipeline.tokenizer.padding_side != 'left':
                self.pipeline.tokenizer.padding_side = 'left'

    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict]]:
        """
        Executes a single chat completion and returns content + logprobs_seq.
        Returns: (generated_text, logprobs_seq)
        logprobs_seq: list of dicts like [{'token': str, 'logprob': float, 'top_logprobs': {str: float}}]
        """
        # Prepare prompts
        prompt_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }
        
        # Handle 'think' parameter for supported models (like Qwen)
        # If the tokenizer doesn't support it, transformers might warn but usually accepts kwargs 
        # (or we can just check if we really need to pass it, but passing it explicitly is safer for Qwen)
        prompt_kwargs["enable_thinking"] = self.think

        try:
            prompt_str = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                **prompt_kwargs
            )
        except TypeError:
            # Simple fallback: if 'enable_thinking' is not supported, remove it
            prompt_kwargs.pop("enable_thinking", None)
            prompt_str = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                **prompt_kwargs
            )

        # Generation Args (Enforced Greedy Decoding)
        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "output_scores": True,  # Critical for LogProbs
            "return_dict_in_generate": True, # Critical for accessing scores
            "do_sample": False, # Enforce Greedy Decoding for classification stability
            "pad_token_id": self.pipeline.tokenizer.pad_token_id,
            "tokenizer": self.pipeline.tokenizer
        }

        inputs = self.pipeline.tokenizer(prompt_str, return_tensors="pt").to(self.pipeline.device)
        
        # Direct generation to get scores
        with torch.no_grad():
            outputs = self.pipeline.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Process Output
        sequences = outputs.sequences
        scores = outputs.scores 
        
        # Extract generated tokens (exclude input prompt)
        input_len = inputs['input_ids'].shape[1]
        generated_ids = sequences[0][input_len:]
        generated_text = self.pipeline.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract per-token logprobs with top-k candidates
        logprobs_seq = []
        if scores:
            for i, token_id in enumerate(generated_ids):
                if i < len(scores):
                    # Step i scores: [Batch=1, Vocab]
                    step_scores = scores[i]
                    # Apply Softmax to get probabilities -> Log -> LogSoftmax
                    step_log_probs = torch.nn.functional.log_softmax(step_scores, dim=-1)
                    # Get log_prob of the selected token
                    token_log_prob = step_log_probs[0, token_id].item()
                    token_str = self.pipeline.tokenizer.decode([token_id], skip_special_tokens=False)
                    
                    # Get top-k logprobs (all vocab in local model)
                    top_k_count = self.config.get("return_logprobs_count", 20)
                    top_k_logprobs_vals, top_k_logprobs_indices = torch.topk(
                        step_log_probs[0], k=min(top_k_count, step_log_probs.shape[-1])
                    )
                    
                    top_logprobs_dict = {}
                    for val, idx in zip(top_k_logprobs_vals.cpu().numpy(), top_k_logprobs_indices.cpu().numpy()):
                        top_token_str = self.pipeline.tokenizer.decode([idx], skip_special_tokens=False)
                        top_logprobs_dict[top_token_str] = float(val)
                    
                    logprobs_seq.append({
                        'token': token_str,
                        'logprob': token_log_prob,
                        'top_logprobs': top_logprobs_dict
                    })

        return generated_text, logprobs_seq
