
import math
import logging
from typing import List, Dict, Tuple
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI
from src.llm_client.base_client import BaseClient

# Setup logger for specific client debug info
logger = logging.getLogger(__name__)

# Suppress Azure HTTP logging (INFO level is too verbose for loop calls)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress OpenAI/httpx logging
logging.getLogger("openai").setLevel(logging.WARNING) # Suppress OpenAI library logging

class AzureGPTClient(BaseClient):
    """
    Client for Azure OpenAI Service.
    Supports native logprobs execution.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 1. Extract Azure-specific configs
        self.api_version = config.get("api_version", "2024-02-15-preview")
        self.azure_endpoint = config.get("azure_endpoint")
        self.model = config.get("model")
        self.scope = config.get("azure_ad_scope", "api://trapi/.default")
        
        # Mapping standard config keys to OpenAI specific keys if needed
        self.max_completion_tokens = config.get("max_completion_tokens", 
                                        config.get("max_tokens", 20))
        
        if not self.azure_endpoint or not self.model:
            raise ValueError("AzureGPTClient requires 'azure_endpoint' and 'model' in config.")

        # 2. Authentication (Trapi Environment Standard)
        try:
            credential = ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            )
            self.token_provider = get_bearer_token_provider(credential, self.scope)
        except Exception as e:
            logger.error(f"Failed to initialize Azure Credentials: {e}")
            raise e

        # 3. Initialize OpenAI Client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_ad_token_provider=self.token_provider,
            api_version=self.api_version,
            max_retries=0 # Retries handled by BaseClient
        )
        
        self.return_logprobs = config.get("return_logprobs", True)
        self.return_logprobs_count = config.get("return_logprobs_count", 20)

    def predict_once(self, messages: List[Dict[str, str]]) -> Tuple[str, List[Dict]]:
        """
        Executes a single chat completion and returns content + logprob.
        """
        
        # Prepare kwargs
        completion_args = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
        }
        
        # Conditionally add logprobs params
        # Note: Some models (o1/gpt-5) might throw 400 if logprobs is passed
        # We try to use it if configured
        if self.return_logprobs:
            completion_args["logprobs"] = True
            completion_args["top_logprobs"] = self.return_logprobs_count

        try:
            response = self.client.chat.completions.create(**completion_args)
            
            choice = response.choices[0]
            content = choice.message.content.strip() if choice.message.content else ""
            
            logprobs_seq = []
            
            # Extract sequence of top logprobs
            if hasattr(choice, 'logprobs') and choice.logprobs and getattr(choice.logprobs, 'content', None):
                tokens = choice.logprobs.content
                for t in tokens:
                    step_dict = {}
                    if hasattr(t, 'top_logprobs') and t.top_logprobs:
                        for tl in t.top_logprobs:
                            step_dict[tl.token] = tl.logprob
                    logprobs_seq.append({
                        'token': t.token,
                        'logprob': t.logprob,
                        'top_logprobs': step_dict
                    })
            
            return content, logprobs_seq

        except Exception as e:
            # Fallback logic for "Unsupported parameter" (e.g. logprobs not supported)
            if "Unsupported parameter" in str(e) and "logprobs" in str(e):
                logger.warning(f"Model {self.model} does not support logprobs. Retrying without logprobs.")
                self.return_logprobs = False
                # User config asks for logprobs but model denies. 
                # We return standard content with dummy confidence.
                # Recursive call without logprobs args
                return self.predict_once(messages)
            
            # Fallback for "Unsupported value: temperature"
            if "Unsupported value" in str(e) and "temperature" in str(e):
                logger.warning(f"Model {self.model} does not support temperature={self.temperature}. Retrying with default.")
                self.temperature = 1 # Set to default for reasoning models
                return self.predict_once(messages)

            raise e

