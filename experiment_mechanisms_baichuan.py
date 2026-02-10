import os
import json
import torch
import argparse
import sys
import logging
import warnings
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import HookManager, calculate_refusal_vector

# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Patch Qwen2VLVisionBlock to handle rotary_pos_emb vs position_embeddings mismatch
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock
    
    def patch_qwen2_vl_vision_block():
        original_forward = Qwen2VLVisionBlock.forward
        
        def patched_forward(self, hidden_states, attention_mask=None, position_embeddings=None, rotary_pos_emb=None, **kwargs):
            # If rotary_pos_emb is provided but position_embeddings is not, map it
            if position_embeddings is None and rotary_pos_emb is not None:
                # Handle tuple/list
                if isinstance(rotary_pos_emb, (list, tuple)):
                    if len(rotary_pos_emb) > 2:
                        position_embeddings = rotary_pos_emb[:2]
                    else:
                        position_embeddings = rotary_pos_emb
                # Handle Tensor
                elif hasattr(rotary_pos_emb, 'shape'): 
                    if rotary_pos_emb.shape[0] > 2:
                        position_embeddings = rotary_pos_emb[:2]
                    else:
                        position_embeddings = rotary_pos_emb
                else:
                    position_embeddings = rotary_pos_emb
            
            # Fix dimension mismatch (e.g. 40 vs 80)
            cos = None
            sin = None
            is_tuple_or_list = False

            if position_embeddings is not None:
                if isinstance(position_embeddings, (list, tuple)) and len(position_embeddings) == 2:
                    cos, sin = position_embeddings
                    is_tuple_or_list = True
                elif hasattr(position_embeddings, 'shape') and position_embeddings.shape[0] == 2:
                    cos = position_embeddings[0]
                    sin = position_embeddings[1]
            
            if cos is not None and sin is not None:
                if hasattr(self, 'attn') and hasattr(cos, 'shape'):
                    head_dim = None
                    if hasattr(self.attn, 'head_dim'):
                        head_dim = self.attn.head_dim
                    elif hasattr(self.attn, 'embed_dim') and hasattr(self.attn, 'num_heads'):
                        head_dim = self.attn.embed_dim // self.attn.num_heads
                    
                    if head_dim is not None and cos.shape[-1] != head_dim:
                        # If 40 -> 80, repeat
                        if head_dim == cos.shape[-1] * 2:
                             cos = torch.cat([cos, cos], dim=-1)
                             sin = torch.cat([sin, sin], dim=-1)
                             if is_tuple_or_list:
                                 position_embeddings = (cos, sin)
                             else:
                                 position_embeddings = torch.stack([cos, sin], dim=0)
            
            # Do NOT pass attention_mask to original_forward as it causes conflict in Qwen2VL
            return original_forward(self, hidden_states, position_embeddings=position_embeddings, rotary_pos_emb=rotary_pos_emb, **kwargs)
            
        Qwen2VLVisionBlock.forward = patched_forward
        print("Patched Qwen2VLVisionBlock.forward to accept rotary_pos_emb")
        
    patch_qwen2_vl_vision_block()
except ImportError:
    print("Could not import Qwen2VLVisionBlock for patching. If you are using Qwen2-VL based models, this might cause issues.")
except Exception as e:
    print(f"Error patching Qwen2VLVisionBlock: {e}")

# Add web_demo to sys.path to import generation if needed
web_demo_path = "/mnt/data/miyan/user_simulation/omni_safety_research/Baichuan-Omni-1.5/web_demo"
if os.path.exists(web_demo_path):
    sys.path.append(web_demo_path)

def load_baichuan_omni_model(model_path, device):
    """
    Loads the Baichuan Omni model and tokenizer onto a specific device.
    """
    print(f"Loading Baichuan-Omni model from {model_path}...")
    # We need to set the cache dir for bind_processor
    cache_dir = os.path.join(os.path.dirname(model_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path=cache_dir)
    print("Model loaded.")
    return model, tokenizer

def run_mechanism_analysis(args):
    model, tokenizer = load_baichuan_omni_model(args.model_path, args.device)
    
    hook_manager = HookManager(model)
    target_layers = list(range(15, 25))
    layer_path = "model.layers"
    
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_path = "model.layers"
    elif hasattr(model, "layers"):
        layer_path = "layers"
    
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    print("Step 1: Calculating Text Refusal Vector...")
    text_activations = {} # {layer_idx: {"safe": [], "harmful": []}}
    
    train_subset = data["text_train"]
    
    for item in tqdm(train_subset, desc="Processing Text Train Data"):
        system_prompt = "You are a helpful assistant."
        text = item["text"]
        input_string = f"<B_SYS>{system_prompt}<C_Q>{text}<C_A>"
        
        # Process input
        ret = model.processor([input_string])
        
        # Move tensors to device
        input_ids = ret.input_ids.to(model.device)
        attention_mask = ret.attention_mask.to(model.device) if ret.attention_mask is not None else None
        
        # Helper to move list of tensors
        def to_device(t_list):
            if t_list is None: return None
            return [t.to(model.device) for t in t_list]

        images = to_device(ret.images)
        videos = to_device(ret.videos)
        audios = ret.audios.to(model.device) if ret.audios is not None else None
        
        # Other args
        patch_nums = ret.patch_nums
        images_grid = ret.images_grid
        videos_patch_nums = ret.videos_patch_nums
        videos_grid = ret.videos_grid
        encoder_length = ret.encoder_length.to(model.device) if ret.encoder_length is not None else None
        bridge_length = ret.bridge_length.to(model.device) if ret.bridge_length is not None else None

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                patch_nums=patch_nums,
                images_grid=images_grid,
                videos=videos,
                videos_patch_nums=videos_patch_nums,
                videos_grid=videos_grid,
                audios=audios,
                encoder_length=encoder_length,
                bridge_length=bridge_length
            )
            
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
    pt_path = os.path.join(args.output_dir, "refusal_vectors_baichuan.pt")
    torch.save(refusal_vectors, pt_path)
    print(f"Refusal vectors saved (torch) to: {pt_path}")

    json_path = os.path.join(args.output_dir, "refusal_vectors_baichuan.json")
    serializable = {str(k): v.cpu().tolist() for k, v in refusal_vectors.items()}
    with open(json_path, "w") as jf:
        json.dump(serializable, jf, indent=2)
    print(f"Refusal vectors saved (json) to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Baichuan-Omni 模型路径")
    parser.add_argument("--data_file", type=str, required=True, help="实验数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="output/mechanisms_baichuan", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    
    args = parser.parse_args()
    
    run_mechanism_analysis(args)
