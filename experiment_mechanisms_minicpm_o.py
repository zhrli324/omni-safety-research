import os
import json
import torch
import argparse
import sys
import logging
import warnings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
import transformers.models.whisper.modeling_whisper as whisper_mod
from utils import HookManager, calculate_refusal_vector

# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# --- Patches for MiniCPM-o ---

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
    print(f"Loading MiniCPM-o model from {model_path}...")
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
    print("Model loaded.")
    return model, tokenizer

def run_mechanism_analysis(args):
    model, tokenizer = load_minicpm_o_model(args.model_path, args.device)
    
    hook_manager = HookManager(model)
    
    layer_path = "llm.model.layers"
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "layers"):
        layer_path = "llm.model.layers"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_path = "model.layers"
    elif hasattr(model, "layers"):
        layer_path = "layers"
    
    target_layers = list(range(8, 18))
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    print("Step 1: Calculating Text Refusal Vector...")
    text_activations = {} # {layer_idx: {"safe": [], "harmful": []}}
    
    train_subset = data["text_train"]
    
    for item in tqdm(train_subset, desc="Processing Text Train Data"):
        text = item["text"]
        
        # Construct inputs
        msgs = [{"role": "user", "content": text}]
        try:
            input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            input_text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
            
        model_inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            # MiniCPM-o forward
            # We use model.llm to bypass the multimodal wrapper's 'data' requirement if possible
            if hasattr(model, "llm"):
                model.llm(**model_inputs)
            else:
                try:
                    model(**model_inputs)
                except TypeError:
                    model(input_ids=model_inputs.input_ids)
            
        for layer_idx in target_layers:
            if layer_idx in hook_manager.activations:
                act = hook_manager.activations[layer_idx][:, -1, :]
                if layer_idx not in text_activations:
                    text_activations[layer_idx] = {"safe": [], "harmful": []}
                text_activations[layer_idx][item["type"]].append(act)
        
        hook_manager.activations = {} # Clear for next batch

    refusal_vectors = {}
    for layer_idx in target_layers:
        if layer_idx in text_activations:
            safe_acts = text_activations[layer_idx]["safe"]
            harm_acts = text_activations[layer_idx]["harmful"]
            if safe_acts and harm_acts:
                refusal_vectors[layer_idx] = calculate_refusal_vector(safe_acts, harm_acts)

    os.makedirs(args.output_dir, exist_ok=True)
    pt_path = os.path.join(args.output_dir, "refusal_vectors_minicpm_o.pt")
    torch.save(refusal_vectors, pt_path)
    print(f"Refusal vectors saved (torch) to: {pt_path}")

    json_path = os.path.join(args.output_dir, "refusal_vectors_minicpm_o.json")
    serializable = {str(k): v.cpu().tolist() for k, v in refusal_vectors.items()}
    with open(json_path, "w") as jf:
        json.dump(serializable, jf, indent=2)
    print(f"Refusal vectors saved (json) to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="MiniCPM-o 模型路径")
    parser.add_argument("--data_file", type=str, required=True, help="实验数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="output/mechanisms_minicpm_o", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    
    args = parser.parse_args()
    
    run_mechanism_analysis(args)
