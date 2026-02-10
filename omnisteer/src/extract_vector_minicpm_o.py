import os
import argparse
import torch
import json
import sys
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
import transformers.models.whisper.modeling_whisper as whisper_mod
from utils import calculate_refusal_vector, SafetyDataset

# --- Patches from unified_eval_minicpm_o.py ---

# Patch DynamicCache.seen_tokens if missing
if not hasattr(DynamicCache, "seen_tokens"):
    try:
        @property
        def seen_tokens(self):
            return self.get_seq_length()
        DynamicCache.seen_tokens = seen_tokens
    except Exception as e:
        print(f"Failed to patch DynamicCache.seen_tokens: {e}")

# Patch WHISPER_ATTENTION_CLASSES if missing
if not hasattr(whisper_mod, "WHISPER_ATTENTION_CLASSES"):
    try:
        attention_classes = {"eager": whisper_mod.WhisperAttention}
        if hasattr(whisper_mod, "WhisperSdpaAttention"):
            attention_classes["sdpa"] = whisper_mod.WhisperSdpaAttention
        else:
            attention_classes["sdpa"] = whisper_mod.WhisperAttention
            
        if hasattr(whisper_mod, "WhisperFlashAttention2"):
            attention_classes["flash_attention_2"] = whisper_mod.WhisperFlashAttention2
        else:
            attention_classes["flash_attention_2"] = whisper_mod.WhisperAttention
            
        whisper_mod.WHISPER_ATTENTION_CLASSES = attention_classes
    except Exception as e:
        print(f"Failed to patch WHISPER_ATTENTION_CLASSES: {e}")

# Patch WhisperAttention.forward
if hasattr(whisper_mod, "WhisperAttention"):
    original_forward = whisper_mod.WhisperAttention.forward
    
    def patched_forward(self, *args, **kwargs):
        try:
            outputs = original_forward(self, *args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                return outputs + (None,)
            return outputs
        except Exception as e:
            if not hasattr(self.config, "_attn_implementation") or self.config._attn_implementation is None:
                self.config._attn_implementation = "eager"
                outputs = original_forward(self, *args, **kwargs)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    return outputs + (None,)
                return outputs
            raise e

    whisper_mod.WhisperAttention.forward = patched_forward

# --- End Patches ---

def load_minicpm_o_model(model_path, device):
    """
    Loads the MiniCPM-o 2.6 model and tokenizer.
    """
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer

def extract_vector(args):
    model, tokenizer = load_minicpm_o_model(args.model_path, args.device)
    
    # Hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            activations[name] = hidden_state.detach()
        return hook

    # Register hook on the target layer
    # MiniCPM-o 2.6 structure
    layers = None
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "layers"):
        layers = model.llm.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        # Fallback search
        for name, module in model.named_modules():
            if name.endswith("layers") and isinstance(module, torch.nn.ModuleList):
                layers = module
                break
    
    if layers is None:
        raise ValueError("Could not find layers in model")

    layer = layers[args.layer_idx]
    layer.register_forward_hook(get_activation(f"layer_{args.layer_idx}"))
    
    dataset = SafetyDataset(args.data_path, tokenizer)
    
    safe_acts = []
    harmful_acts = []
    
    print("Extracting activations...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        text = item.get("prompt", item.get("text", ""))
        
        # Construct inputs
        # Use tokenizer's chat template
        msgs = [{"role": "user", "content": text}]
        
        # MiniCPM-o tokenizer usually handles chat template
        try:
            input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback if apply_chat_template fails or is not configured
            input_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
        model_inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            # MiniCPM-o forward
            # We use model.llm to bypass the multimodal wrapper's 'data' requirement
            if hasattr(model, "llm"):
                model.llm(**model_inputs)
            else:
                # Fallback to original attempt if llm attribute is missing
                try:
                    model(**model_inputs)
                except TypeError:
                    model(input_ids=model_inputs.input_ids)
            
        act = activations[f"layer_{args.layer_idx}"] # [1, seq_len, hidden_dim]
        last_token_act = act[:, -1, :].cpu() # [1, hidden_dim]
        
        if item["type"] == "harmful":
            harmful_acts.append(last_token_act)
        else:
            safe_acts.append(last_token_act)
            
    print(f"Collected {len(safe_acts)} safe and {len(harmful_acts)} harmful activations.")
    
    if not safe_acts or not harmful_acts:
        print("Error: Need both safe and harmful samples.")
        return

    refusal_vector = calculate_refusal_vector(safe_acts, harmful_acts)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(refusal_vector, args.output_path)
    print(f"Refusal vector saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/refusal_vector_minicpm_o.pt")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    extract_vector(args)
