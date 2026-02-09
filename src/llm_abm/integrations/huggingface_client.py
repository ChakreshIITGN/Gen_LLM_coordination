"""HuggingFace client for LLM inference."""

from typing import Optional
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class HuggingFaceClient:
    """Client for querying HuggingFace models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize HuggingFace client.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "microsoft/phi-2")
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        self.model_name = model_name
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading HuggingFace model: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        # Move to device if not using device_map
        if self.device == "cpu" and not (hasattr(self.model, "hf_device_map") and self.model.hf_device_map):
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Convenience method for generation."""
        return self.generate(prompt, **kwargs)
