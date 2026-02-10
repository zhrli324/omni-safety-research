import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

class SafetyAdapter(nn.Module):
    """
    A lightweight trainable tensor that is added to the input embeddings.
    """
    def __init__(self, hidden_size: int, init_scale: float = 1e-3):
        super().__init__()
        # We use a global bias vector that is broadcasted across the sequence length.
        # Shape: (1, 1, hidden_size)
        self.bias = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Initialize with small random values or zeros
        nn.init.normal_(self.bias, mean=0.0, std=init_scale)

    def forward(self, hidden_states):
        # Ensure bias is in the same dtype as hidden_states
        return hidden_states + self.bias.to(hidden_states.dtype)

class SafetyModelWrapper(nn.Module):
    """
    Wraps the Qwen2.5-Omni model to inject the SafetyAdapter.
    """
    def __init__(self, model, adapter: SafetyAdapter):
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """
        Registers a forward_pre_hook on the transformer backbone to inject the bias.
        Target: model.thinker.model (The Qwen2_5OmniModel instance)
        """
        # We need to find the correct submodule. 
        # Based on Qwen2.5-Omni structure: model -> thinker -> model (Transformer)
        if hasattr(self.model, "thinker"):
            target_module = self.model.thinker.model
        else:
            # Fallback if structure is different (e.g. just the thinker loaded)
            target_module = self.model
            
        self.hook_handle = target_module.register_forward_pre_hook(
            self._pre_forward_hook, with_kwargs=True
        )
        print(f"Hook registered on {type(target_module)}")

    def _pre_forward_hook(self, module, args, kwargs):
        """
        Intercepts inputs_embeds and adds the safety bias.
        """
        # Check if inputs_embeds is in kwargs
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            kwargs["inputs_embeds"] = self.adapter(kwargs["inputs_embeds"])
        # Check if inputs_embeds is in args (positional)
        # Qwen2_5OmniModel forward signature: 
        # (input_ids=None, past_key_values=None, attention_mask=None, inputs_embeds=None, ...)
        # It's unlikely to be passed positionally as the 4th arg without others, but we can check.
        elif len(args) > 3 and args[3] is not None:
            # Tuples are immutable, so we would need to reconstruct args
            # This is complex, usually transformers use kwargs for inputs_embeds
            pass
            
        return args, kwargs

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def forward(self, *args, **kwargs):
        # Just pass through to the original model
        # The hook will handle the injection
        return self.model(*args, **kwargs)

    def save_adapter(self, path):
        torch.save(self.adapter.state_dict(), path)

    def load_adapter(self, path):
        self.adapter.load_state_dict(torch.load(path))
