import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import logging
import warnings

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from omni_safety_research.OMNIGUARD.omniguard_core import OmniGuardProcessor, SafetyClassifier, SimpleEmbeddingDataset
from omni_safety_research.OMNIGUARD.custom_data_utils import prepare_calibration_data, prepare_training_data

# Add web_demo to sys.path for Baichuan
web_demo_path = ""
sys.path.append(web_demo_path)

# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ============================
# ============================
MODEL_ID = "baichuan-inc/Baichuan-Omni-1d5"
MODALITY = "vision" # "vision" or "audio"
JSONL_PATH = "data/holisafe_abs.jsonl" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Baichuan Specifics
# ============================
def load_baichuan_omni_model(model_path, device):
    """
    Loads the Baichuan Omni model and tokenizer.
    """
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

def get_special_tokens(model, tokenizer):
    tokens = {}
    # Fallback defaults based on demo
    tokens['video_start'] = '<video_start_baichuan>'
    tokens['video_end'] = '<video_end_baichuan>'
    tokens['image_start'] = '<image_start_baichuan>'
    tokens['image_end'] = '<image_end_baichuan>'
    tokens['audio_start'] = '<audio_start_baichuan>'
    tokens['audio_end'] = '<audio_end_baichuan>'
    return tokens

class BaichuanOmniGuardProcessor(OmniGuardProcessor):
    def __init__(self, model, tokenizer, modality_type="vision"):
        super().__init__(model, tokenizer, modality_type)
        self.special_tokens = get_special_tokens(model, tokenizer)

    def _get_modality_masks(self, input_ids):
        # Not used if we use separate inputs strategy
        raise NotImplementedError("Baichuan uses separate inputs strategy for U-Score.")

def construct_baichuan_input(item, data_dir, special_tokens, only_modality=None):
    """
    Constructs input string.
    only_modality: 'image', 'text', or None (mixed)
    """
    system_prompt = "You are a helpful assistant."
    content = ""
    
    # Image
    if (only_modality is None or only_modality == 'image') and "image" in item:
        image_path = item["image"]
        if not os.path.isabs(image_path):
             image_path = os.path.join(data_dir, image_path)
        abs_image_path = os.path.abspath(image_path)
        content += f"{special_tokens['image_start']}" + json.dumps({"local": abs_image_path}, ensure_ascii=False) + f"{special_tokens['image_end']}"

    # Video
    if (only_modality is None or only_modality == 'image') and "video" in item: # Treat video as image modality
        video_path = item["video"]
        if not os.path.isabs(video_path):
             video_path = os.path.join(data_dir, video_path)
        abs_video_path = os.path.abspath(video_path)
        content += f"{special_tokens['video_start']}" + json.dumps({"local": abs_video_path}, ensure_ascii=False) + f"{special_tokens['video_end']}"

    # Audio
    if (only_modality is None or only_modality == 'audio') and "audio" in item:
        audio_path = item["audio"]
        if not os.path.isabs(audio_path):
             audio_path = os.path.join(data_dir, audio_path)
        abs_audio_path = os.path.abspath(audio_path)
        content += f"{special_tokens['audio_start']}" + json.dumps({"path": abs_audio_path}, ensure_ascii=False) + f"{special_tokens['audio_end']}"

    # Text
    if (only_modality is None or only_modality == 'text') and "text" in item:
        text = item.get("text", "")
        content += text

    full_input = f"<B_SYS>{system_prompt}<C_Q>{content}<C_A>"
    return full_input

def process_baichuan_input(model, input_string):
    ret = model.processor([input_string])
    
    input_ids = ret.input_ids.to(model.device)
    attention_mask = ret.attention_mask.to(model.device) if ret.attention_mask is not None else None
    
    images = [torch.tensor(img, dtype=torch.float32).to(model.device) for img in ret.images] if ret.images is not None else None
    videos = [torch.tensor(img, dtype=torch.float32).to(model.device) for img in ret.videos] if ret.videos is not None else None
    audios = ret.audios.to(model.device) if ret.audios is not None else None
    
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,
        "patch_nums": ret.patch_nums,
        "images_grid": ret.images_grid,
        "videos": videos,
        "videos_patch_nums": ret.videos_patch_nums,
        "videos_grid": ret.videos_grid,
        "audios": audios,
        "encoder_length": ret.encoder_length.to(model.device) if ret.encoder_length is not None else None,
        "bridge_length": ret.bridge_length.to(model.device) if ret.bridge_length is not None else None
    }
    return inputs

def load_and_process_separate_baichuan(data_list, model, special_tokens, modality="vision"):
    """
    Returns list of (image_inputs, text_inputs) tuples.
    """
    pairs = []
    for media_path, text in tqdm(data_list, desc="Loading Separate Inputs"):
        try:
            item = {"text": text}
            if modality == "vision":
                # Check extension
                ext = os.path.splitext(media_path)[1].lower()
                if ext in ['.mp4', '.avi', '.mov']:
                    item["video"] = media_path
                else:
                    item["image"] = media_path
            elif modality == "audio":
                item["audio"] = media_path
            
            # Construct separate inputs
            img_str = construct_baichuan_input(item, "", special_tokens, only_modality=modality) # Only Image/Audio
            txt_str = construct_baichuan_input(item, "", special_tokens, only_modality='text') # Only Text
            
            img_inputs = process_baichuan_input(model, img_str)
            txt_inputs = process_baichuan_input(model, txt_str)
            
            pairs.append((img_inputs, txt_inputs))
            
        except Exception as e:
            print(f"Error processing {media_path}: {e}")
            continue
    return pairs

def load_and_process_batch_baichuan(data_list, model, special_tokens, modality="vision"):
    """
    Returns list of mixed inputs.
    """
    inputs_list = []
    for media_path, text in tqdm(data_list, desc="Loading Mixed Inputs"):
        try:
            item = {"text": text}
            if modality == "vision":
                ext = os.path.splitext(media_path)[1].lower()
                if ext in ['.mp4', '.avi', '.mov']:
                    item["video"] = media_path
                else:
                    item["image"] = media_path
            elif modality == "audio":
                item["audio"] = media_path
            
            full_str = construct_baichuan_input(item, "", special_tokens, only_modality=None) # Mixed
            inputs = process_baichuan_input(model, full_str)
            inputs_list.append(inputs)
            
        except Exception as e:
            print(f"Error processing {media_path}: {e}")
            continue
    return inputs_list

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

def main():
    print(f"Loading model: {MODEL_ID}...")
    model, tokenizer = load_baichuan_omni_model(MODEL_ID, DEVICE)
    special_tokens = get_special_tokens(model, tokenizer)
    
    guard_processor = BaichuanOmniGuardProcessor(model, tokenizer, modality_type=MODALITY)

    # # =======================================================
    # # =======================================================
    # print("\n=== Step 1: Calculating U-Score ===")
    
    # print("Step1: Preparing calibration data...")
    # aligned_list, unaligned_list = prepare_calibration_data(JSONL_PATH, modality=MODALITY, num_samples=50)
    
    # aligned_sims = []
    # unaligned_sims = []

    # print("\nStep2: Processing aligned samples...")
    # aligned_pairs = load_and_process_separate_baichuan(aligned_list, model, special_tokens, modality=MODALITY)
    
    # for idx, (img_in, txt_in) in enumerate(aligned_pairs, start=1):
    #     try:
    #         sims = guard_processor.compute_alignment_score_separate(img_in, txt_in)
    #         aligned_sims.append(sims)
    #     except Exception as e:
    #         import traceback
    #         print(f"    Error computing aligned sample {idx}: {e}")
    #         traceback.print_exc()
    #         continue

    # print("\nStep3: Processing unaligned samples...")
    # unaligned_pairs = load_and_process_separate_baichuan(unaligned_list, model, special_tokens, modality=MODALITY)
    
    # for idx, (img_in, txt_in) in enumerate(unaligned_pairs, start=1):
    #     try:
    #         sims = guard_processor.compute_alignment_score_separate(img_in, txt_in)
    #         unaligned_sims.append(sims)
    #     except Exception as e:
    #         import traceback
    #         print(f"    Error computing unaligned sample {idx}: {e}")
    #         traceback.print_exc()
    #         continue

    # print("\nStep4: Calculating U-Scores...")
    # if aligned_sims and unaligned_sims:
    #     try:
    #         # Stack and pad if lengths differ (shouldn't if model is same)
    #         # Assuming same length
    #         avg_aligned = np.mean(np.stack(aligned_sims, axis=0), axis=0)
    #         avg_unaligned = np.mean(np.stack(unaligned_sims, axis=0), axis=0)
    #         u_scores = avg_aligned - avg_unaligned
            
    #         top_k = min(5, len(u_scores))
    #         print(f"  -> Top {top_k} layers by U-Score:")
    #         ranked = np.argsort(-u_scores)
    #         for i in range(top_k):
    #             layer = ranked[i]
    #             print(f"     Layer {layer}: U-Score={u_scores[layer]:.6f}")
    #         best_layer = int(np.argmax(u_scores))
    #         print(f"--> Best Layer selected: {best_layer} (Max U-Score: {u_scores[best_layer]:.6f})")
    #     except Exception as e:
    #         print(f"Error during U-Score aggregation: {e}")
    #         return
    # else:
    #     print("Error: No valid data for U-Score calculation.")
    #     return

    best_layer = 19

    # =======================================================
    # =======================================================
    print("\n=== Step 2: Extracting Embeddings ===")
    
    harmful_list, safe_list = prepare_training_data(JSONL_PATH, modality=MODALITY)
    
    def extract_features_from_list(data_list):
        # Use Mixed Inputs for Classifier Training
        processed_inputs = load_and_process_batch_baichuan(data_list, model, special_tokens, modality=MODALITY)
        embeddings = []
        for inputs in tqdm(processed_inputs, desc="Extracting Features"):
            emb = guard_processor.get_layer_embeddings(inputs, target_layer=best_layer)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)

    harmful_embs = extract_features_from_list(harmful_list)
    benign_embs = extract_features_from_list(safe_list)
    
    print(f"--> Extracted: Harmful {harmful_embs.shape}, Benign {benign_embs.shape}")
    
    # =======================================================
    # =======================================================
    print("\n=== Step 3: Training Safety Classifier ===")
    
    if len(harmful_embs) == 0 or len(benign_embs) == 0:
        print("Not enough data to train. Exiting.")
        return

    dataset = SimpleEmbeddingDataset(harmful_embs, benign_embs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_dim = harmful_embs.shape[1]
    guard_model = SafetyClassifier(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(guard_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    guard_model.train()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for embs, labels in dataloader:
            embs, labels = embs.to(DEVICE).float(), labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = guard_model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    print("\n--> Training Complete! Guard Model is ready.")
    
    torch.save(guard_model.state_dict(), "omniguard_classifier_baichuan.pt")
    print("Saved classifier to omniguard_classifier_baichuan.pt")

if __name__ == "__main__":
    main()
