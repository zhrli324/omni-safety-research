import torch
import librosa
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

def load_model(model_path: str, device: str = "cuda"):

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


class HookManager:

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}

    def register_activation_hook(self, layer_indices: List[int], layer_attr_path: str = "model.layers"):

        modules = dict(self.model.named_modules())
        
        for layer_idx in layer_indices:
            layer_name = f"{layer_attr_path}.{layer_idx}"
            if layer_name in modules:
                layer = modules[layer_name]
                hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
                self.hooks.append(hook)
                print(f"已注册 Hook 到层: {layer_name}")
            else:
                print(f"警告: 找不到层 {layer_name}")

    def _get_activation_hook(self, layer_idx):
        def hook(model, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def get_activations(self):
        return self.activations

def calculate_refusal_vector(safe_activations: List[torch.Tensor], harmful_activations: List[torch.Tensor]):

    safe_vecs = [act[-1, :].float() for act in safe_activations]
    harm_vecs = [act[-1, :].float() for act in harmful_activations]
    
    mean_safe = torch.stack(safe_vecs).mean(dim=0)
    mean_harm = torch.stack(harm_vecs).mean(dim=0)
    
    return mean_harm - mean_safe
