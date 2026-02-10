import torch
import torch.nn as nn
import os
import json
from typing import List, Optional, Tuple

class OmniGuardMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class OmniGuard:
    def __init__(self, model, layer_idx: int, classifier_path: str = None, device="cuda"):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.classifier = None
        self.activation = None
        self.hook_handle = None
        self.output_dim = 2
        
        # Determine input dimension based on model config
        if hasattr(model.config, "hidden_size"):
            self.input_dim = model.config.hidden_size
        elif hasattr(model.config, "d_model"): # Some models like Whisper/Qwen-Audio might use d_model
            self.input_dim = model.config.d_model
        else:
            # Fallback or try to infer from a dummy run (not implemented here for simplicity)
            self.input_dim = 4096 # Default for many 7B/8B models, but should be careful
            
        if classifier_path and os.path.exists(classifier_path):
            self.load_classifier(classifier_path)
        else:
            print(f"Warning: OmniGuard classifier not found at {classifier_path}. Defense will be disabled until loaded.")

    def load_classifier(self, path):
        # Load checkpoint
        try:
            state_dict = torch.load(path, map_location=self.device)
            # Infer input dim from state dict if possible
            if "fc1.weight" in state_dict:
                self.input_dim = state_dict["fc1.weight"].shape[1]
            elif "model.0.weight" in state_dict:
                self.input_dim = state_dict["model.0.weight"].shape[1]
            
            # Infer output dim
            if "fc3.weight" in state_dict:
                self.output_dim = state_dict["fc3.weight"].shape[0]
            elif "model.4.weight" in state_dict:
                self.output_dim = state_dict["model.4.weight"].shape[0]
            
            self.classifier = OmniGuardMLP(self.input_dim, output_dim=self.output_dim).to(self.device)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print(f"OmniGuard classifier loaded from {path} (Input Dim: {self.input_dim}, Output Dim: {self.output_dim})")
        except Exception as e:
            print(f"Error loading OmniGuard classifier: {e}")

    def _get_layer_module(self):
        # Helper to find the layer module based on model type
        # This needs to be robust across Qwen, Baichuan, MiniCPM
        if hasattr(self.model, "thinker"):
            if hasattr(self.model.thinker, "model") and hasattr(self.model.thinker.model, "layers"):
                return self.model.thinker.model.layers[self.layer_idx]
            elif hasattr(self.model.thinker, "layers"):
                return self.model.thinker.layers[self.layer_idx]
        
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]
        elif hasattr(self.model, "layers"):
            return self.model.layers[self.layer_idx]
        elif hasattr(self.model, "llm") and hasattr(self.model.llm, "model") and hasattr(self.model.llm.model, "layers"):
             # MiniCPM-o structure
            return self.model.llm.model.layers[self.layer_idx]
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer_idx]
        else:
            raise ValueError(f"Could not locate layer {self.layer_idx} in model {type(self.model)}")

    def _hook_fn(self, module, input, output):
        # Output is usually (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Average pooling over tokens: (Batch, Seq, Dim) -> (Batch, Dim)
        # We assume batch size 1 for inference usually, but handle batch
        # Masking is ignored here for simplicity, assuming padding tokens are handled or minimal impact
        self.activation = torch.mean(hidden_states, dim=1).detach()

    def register_hook(self):
        layer_module = self._get_layer_module()
        self.hook_handle = layer_module.register_forward_hook(self._hook_fn)

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
            self.activation = None

    def is_harmful(self, inputs, **kwargs):
        """
        Run a forward pass to check if input is harmful.
        inputs: dict of model inputs (input_ids, images, etc.)
        """
        if self.classifier is None:
            return False

        self.register_hook()
        
        # try:
        with torch.no_grad():
            # Run a single forward pass
            # We need to handle different model signatures
            
            # For Qwen-Omni, we should call model.thinker if available
            if hasattr(self.model, "thinker"):
                self.model.thinker(**inputs)
            # Special handling for MiniCPM-o which might wrap LLM
            elif hasattr(self.model, "llm") and "input_ids" in inputs:
                    # If it's MiniCPM-o, we might want to run the LLM part if inputs are already processed for LLM
                    # But inputs usually come from processor and might contain 'pixel_values' etc.
                    # If we use model(**inputs), it should work.
                    self.model(**inputs)
            else:
                self.model(**inputs)
            
            if self.activation is None:
                print("Warning: OmniGuard hook did not capture activation.")
                return False
            
            # Cast activation to classifier dtype (usually float32)
            activation = self.activation.to(self.classifier.fc1.weight.dtype)
            
            # Run classifier
            logits = self.classifier(activation)
            
            if self.output_dim == 1:
                probs = torch.sigmoid(logits)
                harmful_prob = probs[0, 0].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                harmful_prob = probs[0, 1].item() # Index 1 is harmful
            
            return harmful_prob > 0.5
                
        # except Exception as e:
        #     print(f"OmniGuard check failed: {e}")
        #     return False
        # finally:
        #     self.remove_hook()

