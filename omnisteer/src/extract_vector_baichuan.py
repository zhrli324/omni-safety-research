import os
import argparse
import torch
import json
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import calculate_refusal_vector, SafetyDataset
import logging
import warnings

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

# Add web_demo to sys.path to import generation if needed, though we might not need it for just loading
# But Baichuan-Omni might rely on it for some internal imports in remote code
web_demo_path = ""
if os.path.exists(web_demo_path):
    sys.path.append(web_demo_path)

def load_baichuan_omni_model(model_path, device):
    """
    Loads the Baichuan Omni model and tokenizer onto a specific device.
    """
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
    
    return model, tokenizer

def extract_vector(args):
    model, tokenizer = load_baichuan_omni_model(args.model_path, args.device)
    model.eval()
    
    # Hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # output is usually a tuple (hidden_states, ...)
            # We take the last token's hidden state
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            activations[name] = hidden_state.detach()
        return hook

    # Register hook on the target layer
    # Baichuan usually has model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[args.layer_idx]
    elif hasattr(model, "layers"):
        layer = model.layers[args.layer_idx]
    else:
        raise ValueError("Could not find layers in model")

    layer.register_forward_hook(get_activation(f"layer_{args.layer_idx}"))
    
    # Pass tokenizer as processor
    dataset = SafetyDataset(args.data_path, tokenizer)
    
    safe_acts = []
    harmful_acts = []
    
    print("Extracting activations...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        # Construct inputs for Baichuan Omni (Text Only)
        # Format: <B_SYS>{system_prompt}<C_Q>{content}<C_A>
        system_prompt = "You are a helpful assistant."
        
        # SafetyDataset usually returns item with "prompt" or "text"
        # Adjust based on what SafetyDataset returns. 
        # The qwen script used item["prompt"].
        text = item.get("prompt", item.get("text", ""))
        
        input_string = f"<B_SYS>{system_prompt}<C_Q>{text}<C_A>"
        
        # Process input
        ret = model.processor([input_string])
        
        # Move tensors to device
        input_ids = ret.input_ids.to(model.device)
        attention_mask = ret.attention_mask.to(model.device) if ret.attention_mask is not None else None
        
        # For text-only, other multimodal args are None or empty
        # We pass them as is from ret, moving to device if present
        
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
            # Run forward pass
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
    parser.add_argument("--output_path", type=str, default="output/refusal_vector_baichuan.pt")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    extract_vector(args)
