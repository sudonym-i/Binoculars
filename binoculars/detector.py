"""Binoculars AI-generated text detector.

This module implements the Binoculars method for detecting AI-generated text using
two language models: an 'observer' model and a 'performer' model. The detection
works by comparing perplexity and cross-perplexity scores between the models.
"""

from typing import Union

import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy

# Disable gradient computation for inference-only operations
torch.set_grad_enabled(False)

# Configuration for accessing private Hugging Face models
huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# Predefined thresholds selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

# Device configuration for multi-GPU setup
DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1


class Binoculars(object):
    """Binoculars AI-generated text detector.
    
    Uses two language models (observer and performer) to compute perplexity-based
    scores for detecting AI-generated text. The detection is based on comparing
    the perplexity of text under both models.
    
    Models are cached at the class level to avoid reloading for multiple instances
    with the same model configuration.
    
    Args:
        observer_name_or_path: HuggingFace model name or path for the observer model
        performer_name_or_path: HuggingFace model name or path for the performer model  
        use_bfloat16: Whether to use bfloat16 precision for faster inference
        max_token_observed: Maximum number of tokens to process per text
        mode: Detection threshold mode, either 'low-fpr' or 'accuracy'
    """
    
    # Class-level cache for loaded models
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self,
                 observer_name_or_path: str = "tiiuae/falcon-7b",
                 performer_name_or_path: str = "tiiuae/falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        # Ensure both models use compatible tokenizers
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        # Set detection threshold based on mode
        self.change_mode(mode)
        
        # Store model configuration
        self.observer_name_or_path = observer_name_or_path
        self.performer_name_or_path = performer_name_or_path
        self.use_bfloat16 = use_bfloat16
        self.max_token_observed = max_token_observed
        
        # Load or retrieve cached models
        self.observer_model = self._get_or_load_model(
            observer_name_or_path, DEVICE_1, use_bfloat16, "observer"
        )
        self.performer_model = self._get_or_load_model(
            performer_name_or_path, DEVICE_2, use_bfloat16, "performer"
        )
        
        # Load or retrieve cached tokenizer
        self.tokenizer = self._get_or_load_tokenizer(observer_name_or_path)

    def _get_or_load_model(self, model_name_or_path: str, device: str, use_bfloat16: bool, model_type: str):
        """Load model from cache or create new instance if not cached."""
        cache_key = f"{model_name_or_path}_{device}_{use_bfloat16}_{model_type}"
        
        if cache_key not in self._model_cache:
            print(f"Loading {model_type} model: {model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map={"": device},
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                token=huggingface_config["TOKEN"]
            )
            model.eval()
            self._model_cache[cache_key] = model
            print(f"✓ {model_type} model loaded and cached")
        else:
            print(f"✓ Using cached {model_type} model: {model_name_or_path}")
            
        return self._model_cache[cache_key]
    
    def _get_or_load_tokenizer(self, model_name_or_path: str):
        """Load tokenizer from cache or create new instance if not cached."""
        if model_name_or_path not in self._tokenizer_cache:
            print(f"Loading tokenizer: {model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer_cache[model_name_or_path] = tokenizer
            print(f"✓ Tokenizer loaded and cached")
        else:
            print(f"✓ Using cached tokenizer: {model_name_or_path}")
            
        return self._tokenizer_cache[model_name_or_path]

    @classmethod
    def clear_cache(cls):
        """Clear all cached models and tokenizers to free memory."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        print("✓ Model cache cleared")

    def change_mode(self, mode: str) -> None:
        """Change the detection threshold mode.
        
        Args:
            mode: Detection mode, either 'low-fpr' (optimized for low false positive rate)
                 or 'accuracy' (optimized for balanced accuracy/F1 score)
                 
        Raises:
            ValueError: If mode is not 'low-fpr' or 'accuracy'
        """
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        """Tokenize a batch of text strings.
        
        Args:
            batch: List of text strings to tokenize
            
        Returns:
            BatchEncoding object containing tokenized inputs ready for model inference
        """
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> tuple[torch.Tensor, torch.Tensor]:
        """Get logits from both observer and performer models.
        
        Args:
            encodings: Tokenized input batch
            
        Returns:
            Tuple of (observer_logits, performer_logits) - raw model outputs
            before softmax, used for computing perplexity and cross-perplexity
        """
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        """Compute Binoculars detection scores for input text(s).
        
        The Binoculars score is calculated as the ratio of perplexity under the
        performer model to cross-perplexity between observer and performer models.
        Lower scores indicate higher likelihood of AI generation.
        
        Args:
            input_text: Single text string or list of text strings to analyze
            
        Returns:
            Single float score (if input is string) or list of float scores
            (if input is list). Lower scores indicate higher probability of AI generation.
        """
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        
        # Calculate perplexity under performer model
        ppl = perplexity(encodings, performer_logits)
        
        # Calculate cross-perplexity between observer and performer models
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        
        # Binoculars score is the ratio of perplexity to cross-perplexity
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        """Make AI vs human-generated predictions for input text(s).
        
        Uses the current threshold (set by mode) to classify text as either
        AI-generated or human-generated based on Binoculars scores.
        
        Args:
            input_text: Single text string or list of text strings to classify
            
        Returns:
            Single prediction string (if input is string) or list of prediction strings
            (if input is list). Each prediction is either "Most likely AI-generated"
            or "Most likely human-generated".
        """
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred
