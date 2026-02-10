import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model, HookManager, calculate_refusal_vector
from qwen_omni_utils import process_mm_info

def run_projection_analysis(args):
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)
    
    hook_manager = HookManager(model)
    target_layers = list(range(args.start_layer, args.end_layer))
    layer_path = "thinker.model.layers" 
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    print("Calculating Text Refusal Vectors...")
    text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
    
    train_subset = data["text_train"]
    if args.max_samples > 0:
        import random
        random.seed(42)
        safe_samples = [item for item in train_subset if item["type"] == "safe"]
        harmful_samples = [item for item in train_subset if item["type"] == "harmful"]
        train_subset = random.sample(safe_samples, min(len(safe_samples), args.max_samples)) + \
                       random.sample(harmful_samples, min(len(harmful_samples), args.max_samples))

    for item in tqdm(train_subset, desc="Text Data"):
        content = [{"type": "text", "text": item["text"]}]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": content}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
            
        for layer in target_layers:
            act = hook_manager.activations[layer][:, -1, :].cpu()
            text_acts[layer][item["type"]].append(act)
        hook_manager.activations = {}

    text_vectors = {}
    for layer in target_layers:
        if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
            text_vectors[layer] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    print("Calculating Projections on Canonical Vectors...")
    results = {}
    
    for layer in target_layers:
        if layer not in text_vectors:
            continue
            
        canon_path = os.path.join(args.canonical_dir, f"canonical_refusal_vector_layer_{layer}.pt")
        if not os.path.exists(canon_path):
            print(f"Warning: Canonical vector for layer {layer} not found at {canon_path}")
            continue
            
        canon_vec = torch.load(canon_path, map_location="cpu").float()
        text_vec = text_vectors[layer].float()
        
        cos_sim = torch.dot(text_vec, canon_vec) / (torch.norm(text_vec) * torch.norm(canon_vec) + 1e-8)
        
        # proj = |text_vec| * cos(theta) = (text_vec . canon_vec) / |canon_vec|
        scalar_proj = torch.dot(text_vec, canon_vec) / (torch.norm(canon_vec) + 1e-8)
        
        proj_ratio = scalar_proj / (torch.norm(canon_vec) + 1e-8)
        
        results[layer] = {
            "refusal_vector": {
                "cosine_similarity": cos_sim.item(),
                "scalar_projection": scalar_proj.item(),
                "projection_ratio": proj_ratio.item(),
                "vector_norm": torch.norm(text_vec).item(),
            },
            "canonical_vector_norm": torch.norm(canon_vec).item()
        }
        
        # --- NEW: Calculate projections for individual harmful and safe hidden states ---
        for label in ["safe", "harmful"]:
            label_acts = text_acts[layer][label]
            if not label_acts:
                continue
            
            # Stack activations: (N, dim)
            X_label = torch.cat(label_acts, dim=0).float()
            
            # Calculate dot products with canonical vector
            dots = torch.mv(X_label, canon_vec)
            norms = torch.norm(X_label, dim=1)
            canon_norm = torch.norm(canon_vec)
            
            cos_sims = dots / (norms * canon_norm + 1e-8)
            scalar_projs = dots / (canon_norm + 1e-8)
            proj_ratios = scalar_projs / (canon_norm + 1e-8)
            
            results[layer][f"{label}_stats"] = {
                "mean_cosine_similarity": cos_sims.mean().item(),
                "std_cosine_similarity": cos_sims.std().item(),
                "mean_scalar_projection": scalar_projs.mean().item(),
                "std_scalar_projection": scalar_projs.std().item(),
                "mean_projection_ratio": proj_ratios.mean().item(),
                "mean_norm": norms.mean().item()
            }
        
        print(f"Layer {layer}: Refusal CosSim = {cos_sim.item():.4f}, Scalar Proj = {scalar_proj.item():.4f}, Ratio = {proj_ratio.item():.4f}")
        if "harmful_stats" in results[layer]:
            h_stats = results[layer]["harmful_stats"]
            print(f"  Harmful Mean CosSim: {h_stats['mean_cosine_similarity']:.4f}, Mean Proj: {h_stats['mean_scalar_projection']:.4f}, Mean Ratio: {h_stats['mean_projection_ratio']:.4f}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--canonical_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="output/projection_results.json")
    parser.add_argument("--start_layer", type=int, default=15)
    parser.add_argument("--end_layer", type=int, default=25)
    parser.add_argument("--max_samples", type=int, default=520, help="Number of samples per class (safe/harmful) to use")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_projection_analysis(args)
