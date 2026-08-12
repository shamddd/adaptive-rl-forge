"""
Pretrained Open Causal Language Model wrapper with PEFT (LoRA) integration.
Supports loading HuggingFace models (e.g. SmolLM-135M, gpt2, Qwen2.5-0.5B).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class PretrainedLM(nn.Module):
    """
    Wrapper for pretrained HuggingFace Causal LMs with optional LoRA adapter.
    """

    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device or torch.device("cpu")

        print(f"Loading pretrained model: {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        if use_lora:
            target_modules = ["c_attn", "c_proj", "q_proj", "v_proj"]
            # Filter target modules matching model
            model_modules = [name for name, _ in base_model.named_modules()]
            valid_targets = [m for m in target_modules if any(m in mod for mod in model_modules)]
            if not valid_targets:
                valid_targets = ["c_attn"]

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=valid_targets,
            )
            self.model = get_peft_model(base_model, peft_config)
            print("LoRA adapter attached successfully.")
        else:
            self.model = base_model

        self.model.to(self.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        self.model.eval()
        outputs = self.model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return outputs
