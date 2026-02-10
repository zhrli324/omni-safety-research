import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import HookManager, calculate_refusal_vector

# Baichuan-Omni helpers (adapted from safe_eval/unified_eval_baichuan.py)
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def load_baichuan_omni_model(model_path: str, device: str):
    cache_dir = os.path.join(os.path.dirname(model_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path=cache_dir)
    return model, tokenizer

def get_special_tokens(model, tokenizer):
    tokens = {}
    if hasattr(model.config, "video_config"):
        tokens["video_start"] = tokenizer.convert_ids_to_tokens(model.config.video_config.video_start_token_id)
        tokens["video_end"] = tokenizer.convert_ids_to_tokens(model.config.video_config.video_end_token_id)
        tokens["image_start"] = tokenizer.convert_ids_to_tokens(model.config.video_config.image_start_token_id)
        tokens["image_end"] = tokenizer.convert_ids_to_tokens(model.config.video_config.image_end_token_id)
    if hasattr(model.config, "audio_config"):
        tokens["audio_start"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
        tokens["audio_end"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
        tokens["audiogen_start"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audiogen_start_token_id)

    tokens.setdefault("video_start", "<video_start_baichuan>")
    tokens.setdefault("video_end", "<video_end_baichuan>")
    tokens.setdefault("image_start", "<image_start_baichuan>")
    tokens.setdefault("image_end", "<image_end_baichuan>")
    tokens.setdefault("audio_start", "<audio_start_baichuan>")
    tokens.setdefault("audio_end", "<audio_end_baichuan>")
    return tokens

def _resolve_path(p: str, base_dir: str | None):
    if p is None:
        return None
    if base_dir and not os.path.isabs(p):
        return os.path.abspath(os.path.join(base_dir, p))
    return os.path.abspath(p)

def construct_baichuan_conversation(item: dict, special_tokens: dict, base_dir: str | None = None, system_prompt: str | None = None):
    system_prompt = system_prompt or "You are a helpful assistant."
    content = ""
    modality = item.get("modality", "")

    img_path = item.get("image_path") or item.get("image")
    img_path = _resolve_path(img_path, base_dir)
    if img_path and "image" in modality:
        content += f"{special_tokens['image_start']}" + json.dumps({"local": img_path}, ensure_ascii=False) + f"{special_tokens['image_end']}"

    vid_path = item.get("video_path") or item.get("video")
    vid_path = _resolve_path(vid_path, base_dir)
    if vid_path and "video" in modality:
        content += f"{special_tokens['video_start']}" + json.dumps({"local": vid_path}, ensure_ascii=False) + f"{special_tokens['video_end']}"

    aud_path = item.get("audio_path") or item.get("audio")
    aud_path = _resolve_path(aud_path, base_dir)
    if aud_path and "audio" in modality:
        content += f"{special_tokens['audio_start']}" + json.dumps({"path": aud_path}, ensure_ascii=False) + f"{special_tokens['audio_end']}"

    text_field = item.get("text_instruction") or item.get("text", "")
    content += text_field

    full_input = f"<B_SYS>{system_prompt}<C_Q>{content}<C_A>"
    return full_input

def prepare_model_inputs(model, processor_outputs):
    inputs = {"input_ids": processor_outputs.input_ids.to(model.device)}
    if processor_outputs.attention_mask is not None:
        inputs["attention_mask"] = processor_outputs.attention_mask.to(model.device)

    # Optional multimodal pieces
    if getattr(processor_outputs, "images", None) is not None:
        inputs["images"] = [torch.tensor(img, dtype=torch.float32).to(model.device) for img in processor_outputs.images]
    if getattr(processor_outputs, "videos", None) is not None:
        inputs["videos"] = [torch.tensor(v, dtype=torch.float32).to(model.device) for v in processor_outputs.videos]
    if getattr(processor_outputs, "audios", None) is not None:
        inputs["audios"] = processor_outputs.audios.to(model.device)

    for key in ["patch_nums", "images_grid", "videos_patch_nums", "videos_grid", "encoder_length", "bridge_length"]:
        if getattr(processor_outputs, key, None) is not None:
            val = getattr(processor_outputs, key)
            inputs[key] = val.to(model.device) if hasattr(val, "to") else val
    return inputs

def select_layer_path(model):
    candidates = ["model.model.layers", "model.layers"]
    modules = dict(model.named_modules())
    for path in candidates:
        if f"{path}.0" in modules:
            return path
    raise ValueError("Could not locate transformer layers for hook registration.")

def run_vector_analysis(args):
    print(f"Loading Baichuan-Omni model from {args.model_path}...")
    model, tokenizer = load_baichuan_omni_model(args.model_path, args.device)
    special_tokens = get_special_tokens(model, tokenizer)

    hook_manager = HookManager(model)
    target_layers = list(range(10, 21))
    layer_path = select_layer_path(model)
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    with open(args.data_file, "r") as f:
        data = json.load(f)

    vectors = {layer: {} for layer in target_layers}

    # Text refusal vectors
    print("Calculating Text Refusal Vectors...")
    text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
    train_subset = data["text_train"]

    for item in tqdm(train_subset, desc="Text Data"):
        convo = {"text": item["text"], "modality": "text"}
        full_input = construct_baichuan_conversation(
            convo,
            special_tokens,
            base_dir=args.data_root,
            system_prompt="You are a helpful assistant.",
        )
        processed = model.processor([full_input])
        model_inputs = prepare_model_inputs(model, processed)

        with torch.no_grad():
            model(**model_inputs)

        for layer in target_layers:
            act = hook_manager.activations[layer][:, -1, :].cpu()
            text_acts[layer][item["type"]].append(act)
        hook_manager.activations = {}

    for layer in target_layers:
        if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
            vectors[layer]["text"] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    # Multi-modal refusal vectors
    target_modalities = ["image", "audio", "video", "text+image", "text+audio", "text+video"]
    mm_acts = {m: {layer: {"safe": [], "harmful": []} for layer in target_layers} for m in target_modalities}
    max_samples = 200
    counts = {m: {"safe": 0, "harmful": 0} for m in target_modalities}

    print("Calculating Multi-Modal Refusal Vectors...")
    for item in tqdm(data["cross_modal_test"], desc="MM Data"):
        modality = item.get("modality", "")
        label = item.get("label", "harmful")
        if modality not in target_modalities:
            continue
        if counts[modality][label] >= max_samples:
            continue

        full_input = construct_baichuan_conversation(
            item,
            special_tokens,
            base_dir=args.data_root,
            system_prompt="You are a helpful assistant.",
        )
        processed = model.processor([full_input])
        model_inputs = prepare_model_inputs(model, processed)

        with torch.no_grad():
            model(**model_inputs)

        for layer in target_layers:
            act = hook_manager.activations[layer][:, -1, :].cpu()
            mm_acts[modality][layer][label].append(act)
        hook_manager.activations = {}
        counts[modality][label] += 1

    for m in target_modalities:
        for layer in target_layers:
            s_acts = mm_acts[m][layer]["safe"]
            h_acts = mm_acts[m][layer]["harmful"]
            if s_acts and h_acts:
                vectors[layer][m] = calculate_refusal_vector(s_acts, h_acts).cpu()

    # Analysis and visualization
    print("Analyzing vectors...")
    os.makedirs(args.output_dir, exist_ok=True)
    analysis_results = {}

    for layer in target_layers:
        layer_vecs = vectors[layer]
        if len(layer_vecs) < 1:
            continue

        base_vec = layer_vecs.get("text")
        aligned_vecs = []
        modality_names = list(layer_vecs.keys())

        if base_vec is not None:
            base_norm = torch.norm(base_vec) + 1e-6
            for m in modality_names:
                v = layer_vecs[m]
                _ = torch.dot(v.to(base_vec.dtype), base_vec) / (torch.norm(v) * base_norm)
                aligned_vecs.append(v)
        else:
            aligned_vecs = [layer_vecs[m] for m in modality_names]

        avg_norm = np.mean([torch.norm(v).item() for v in aligned_vecs])
        X = torch.stack(aligned_vecs).float().numpy()

        pca = PCA(n_components=min(len(modality_names), 3))
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_

        pc1 = pca.components_[0]
        avg_vec = X.mean(axis=0)
        if np.dot(pc1, avg_vec) < 0:
            pc1 = -pc1
        pc1_scaled = pc1 * avg_norm

        pc1_tensor = torch.tensor(pc1_scaled, dtype=torch.float32)
        torch.save(pc1_tensor, os.path.join(args.output_dir, f"pca_refusal_vector_layer_{layer}.pt"))

        pure_keys = ["text", "image", "audio", "video"]
        pure_indices = [i for i, m in enumerate(modality_names) if m in pure_keys]

        if len(pure_indices) > 0:
            pure_aligned_vecs = [aligned_vecs[i] for i in pure_indices]
            mean_vec = torch.stack(pure_aligned_vecs).mean(dim=0)
            torch.save(mean_vec, os.path.join(args.output_dir, f"mean_refusal_vector_layer_{layer}.pt"))

            if "text" in layer_vecs:
                text_vec = layer_vecs["text"]
                sim_mean = torch.dot(mean_vec.to(text_vec.dtype), text_vec) / (torch.norm(mean_vec) * torch.norm(text_vec))
                print(f"Layer {layer}: Pure Mean Vector vs Text CosSim = {sim_mean.item():.4f}")

        if len(pure_indices) > 0:
            X_pure = X[pure_indices]
            _, _, vh = np.linalg.svd(X_pure, full_matrices=False)
            canonical_vec = vh[0]
            if np.dot(canonical_vec, X_pure.mean(axis=0)) < 0:
                canonical_vec = -canonical_vec
            pure_avg_norm = np.mean([torch.norm(layer_vecs[modality_names[i]]).item() for i in pure_indices])
            canonical_vec_scaled = canonical_vec * pure_avg_norm
            canon_tensor = torch.tensor(canonical_vec_scaled, dtype=torch.float32)
            torch.save(canon_tensor, os.path.join(args.output_dir, f"canonical_refusal_vector_layer_{layer}.pt"))

            if "text" in layer_vecs:
                text_vec = layer_vecs["text"].numpy()
                sim = np.dot(canonical_vec, text_vec) / (np.linalg.norm(canonical_vec) * np.linalg.norm(text_vec))
                print(f"Layer {layer}: Canonical (Pure SVD) vs Text CosSim = {sim:.4f}")

        alignment_scores = {}
        for i, m in enumerate(modality_names):
            vec = X[i]
            cos_sim = np.dot(vec, pc1) / (np.linalg.norm(vec) * np.linalg.norm(pc1))
            alignment_scores[m] = float(cos_sim)

        analysis_results[layer] = {
            "explained_variance_ratio": explained_variance.tolist(),
            "pc1_alignment": alignment_scores,
        }

        print(f"Layer {layer}: PC1 Explained Var = {explained_variance[0]:.4f}")

        if pca.n_components_ >= 2:
            X_pca = pca.transform(X)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=modality_names, s=100, style=modality_names)
            for i, txt in enumerate(modality_names):
                plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=12)
                plt.arrow(0, 0, X_pca[i, 0], X_pca[i, 1], color="gray", alpha=0.3, width=0.002)
            plt.title(f"Layer {layer} Refusal Vectors PCA")
            plt.xlabel(f"PC1 ({explained_variance[0]:.2%})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.2%})")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, f"layer_{layer}_pca.png"))
            plt.close()

    with open(os.path.join(args.output_dir, "vector_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Analysis done. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/vector_analysis_baichuan")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=None, help="Optional base dir to resolve relative media paths")
    args = parser.parse_args()
    run_vector_analysis(args)
