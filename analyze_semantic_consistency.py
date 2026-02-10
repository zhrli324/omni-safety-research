import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from utils import load_model, HookManager
from qwen_omni_utils import process_mm_info
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
import pickle
import seaborn as sns


def is_harmful_item(item):
    """Best-effort harmful label detection across common schemas."""
    # Common keys seen in this repo: label (safe/harmful), type (safe/harmful)
    label = item.get("label", None)
    if isinstance(label, str):
        return label.lower() == "harmful"
    label = item.get("type", None)
    if isinstance(label, str):
        return label.lower() == "harmful"
    # Some datasets may use boolean flags
    if isinstance(item.get("is_harmful", None), bool):
        return item["is_harmful"]
    return False

def prepare_inputs(processor, item, device):
    modality = item.get("modality", "text")
    content = []

    # Text Input
    # Some datasets use "text" for pure text, others might use "text_instruction" for MM prompts
    if modality == "text":
        text_content = item.get("text") or item.get("text_instruction") or ""
        content.append({"type": "text", "text": text_content})
    else:
        # Multi-modal Input construction
        if "image" in modality and item.get("image_path"):
            content.append({"type": "image", "image": item["image_path"]})
        if "video" in modality and item.get("video_path"):
            content.append({"type": "video", "video": item["video_path"]})
        if "audio" in modality and item.get("audio_path"):
            content.append({"type": "audio", "audio": item["audio_path"]})
            
        # Check if there is text accompanying the media (e.g. text+image)
        # Note: Pure "image" modality in this context means text rendered on image, 
        # so prompt might not have explicit text, valid for some models. 
        # But if dataset has 'text_instruction', we include it.
        if "text" in modality:
            text_field = item.get("text_instruction") or item.get("text")
            if text_field:
                content.append({"type": "text", "text": text_field})

    conversation = [
        # {
        #     "role": "system",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        #         }
        #     ]
        # },
        {"role": "user", "content": content}
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, False)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def get_last_token_hidden_states(model, hook_manager, inputs, target_layers):
    with torch.no_grad():
        model.thinker(**inputs)
    
    states = {}
    for layer in target_layers:
        # Get activations: [batch, seq, dim] -> [dim] for last token
        if layer in hook_manager.activations:
            # Assume batch size 1
            act = hook_manager.activations[layer][0, -1, :].detach().cpu()
            states[layer] = act
    
    # Clear hooks for next pass
    hook_manager.activations = {}
    return states

def analyze_similarity(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cache_path = os.path.join(args.output_dir, "analysis_cache.pkl")
    
    similarity_stats = None
    tsne_bank = None
    loaded_from_cache = False
    
    # Try loading from cache
    if os.path.exists(cache_path) and not args.force_recompute:
        print(f"Loading cached data from {cache_path}...")
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                similarity_stats_raw = data.get("similarity_stats", {})
                tsne_bank_raw = data.get("tsne_bank", {})
                
                # Reconstruct defaultdicts
                similarity_stats = defaultdict(lambda: defaultdict(list))
                for m, ld in similarity_stats_raw.items():
                    for l, v in ld.items():
                        similarity_stats[m][l] = v
                        
                tsne_bank = defaultdict(lambda: defaultdict(list))
                for l, md in tsne_bank_raw.items():
                    for m, v in md.items():
                        tsne_bank[l][m] = v
                
                loaded_from_cache = True
        except Exception as e:
            print(f"Failed to load cache: {e}. Will recompute.")
            
    if not loaded_from_cache:
        print(f"Loading model from {args.model_path}...")
        model, processor = load_model(args.model_path, args.device)
        
        # Identify layers
        if hasattr(model, "thinker"):
            num_layers = len(model.thinker.model.layers)
            layer_prefix = "thinker.model.layers"
        elif hasattr(model, "model"):
            num_layers = len(model.model.layers)
            layer_prefix = "model.layers"
        else:
            # Fallback/Guess
            num_layers = 32
            layer_prefix = "thinker.model.layers"
            
        print(f"Detected {num_layers} layers.")
        target_layers = list(range(num_layers))
        
        hook_manager = HookManager(model)
        hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_prefix)
        
        print(f"Loading data from {args.data_file}...")
        with open(args.data_file, 'r') as f:
            raw_data = json.load(f)
            
        # Standardize data structure: Assume it's a list or dict with 'text_train', 'cross_modal_test'
        data_list = []
        if isinstance(raw_data, list):
            data_list = raw_data
        elif isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, list):
                    data_list.extend(v)
                elif isinstance(v, dict):
                     # edge case
                     pass
                     
        # Group by ID
        # We assume items with same "id" field are variations of the same prompt
        grouped = defaultdict(dict)
        for item in data_list:
            if "id" in item:
                modality = item.get("modality", "text")
                grouped[item["id"]][modality] = item
                
        print(f"Found {len(grouped)} unique IDs.")
        
        similarity_stats = defaultdict(lambda: defaultdict(list)) # results[modality][layer] = [sims]
        # tsne_bank[layer][modality] = [vectors]
        tsne_bank = defaultdict(lambda: defaultdict(list))
        
        total_ids = len(grouped)
        processed_count = 0
        
        for item_id, variants in tqdm(grouped.items(), total=total_ids, desc="Comparing Modalities"):
            if "text" not in variants:
                continue # Need text as anchor
                
            text_item = variants["text"]
            
            # 1. Get Text Hidden States (Anchor)
            try:
                text_inputs = prepare_inputs(processor, text_item, args.device)
                text_states = get_last_token_hidden_states(model, hook_manager, text_inputs, target_layers)
            except Exception as e:
                # print(f"Error processing text for ID {item_id}: {e}")
                continue

            # Collect TEXT vectors for t-SNE (now regardless of harmful flag)
            if args.do_tsne:
                for layer, vec in text_states.items():
                    if args.tsne_layers_all or layer in args.tsne_layers:
                        if len(tsne_bank[layer]["text"]) < args.tsne_samples:
                            tsne_bank[layer]["text"].append(vec.to(torch.float32).numpy())
                
            # 2. Compare other modalities against text
            for modality, item in variants.items():
                if modality == "text":
                    continue
                    
                try:
                    mm_inputs = prepare_inputs(processor, item, args.device)
                    mm_states = get_last_token_hidden_states(model, hook_manager, mm_inputs, target_layers)
                    
                    for layer in target_layers:
                        if layer in text_states and layer in mm_states:
                            vec_text = text_states[layer].to(torch.float32)
                            vec_mm = mm_states[layer].to(torch.float32)
                            
                            sim = F.cosine_similarity(vec_text.unsqueeze(0), vec_mm.unsqueeze(0)).item()
                            similarity_stats[modality][layer].append(sim)

                            # Collect vectors for t-SNE if enabled
                            if args.do_tsne:
                                if args.tsne_layers_all or layer in args.tsne_layers:
                                    if len(tsne_bank[layer][modality]) < args.tsne_samples:
                                        tsne_bank[layer][modality].append(vec_mm.numpy())
                            
                except Exception as e:
                    # print(f"Error processing {modality} for ID {item_id}: {e}")
                    pass
            
            processed_count += 1
            if args.max_samples and processed_count >= args.max_samples:
                break
        
        # Save cache
        print(f"Saving data to cache at {cache_path}...")
        # Convert to normal dicts before pickling to avoid recursion issues with lambdas
        sim_stats_dict = {k: dict(v) for k, v in similarity_stats.items()}
        tsne_bank_dict = {k: dict(v) for k, v in tsne_bank.items()}
        with open(cache_path, "wb") as f:
            pickle.dump({
                "similarity_stats": sim_stats_dict,
                "tsne_bank": tsne_bank_dict
            }, f)
            
    # Aggregate results: Mean, Std, Count, rough vMF kappa proxy per layer per modality
    aggregated_results = {}
    for modality, layers_data in similarity_stats.items():
        aggregated_results[modality] = {}
        for layer, sims in layers_data.items():
            if sims:
                mean_cos = float(np.mean(sims))
                std_cos = float(np.std(sims))
                count = len(sims)
                # Rough kappa proxy: inverse of (1 - mean_cos), avoid div0
                kappa_proxy = float(1.0 / (1.0 - mean_cos + 1e-6))
                aggregated_results[modality][layer] = {
                    "mean_cosine": mean_cos,
                    "std_cosine": std_cos,
                    "count": count,
                    "kappa_proxy": kappa_proxy
                }
                
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "semantic_consistency_metrics.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)
        
    print(f"Analysis complete. Results saved to {output_path}")
    
    # Optional: Plotting
    if args.plot:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.rcParams.update({
                "font.family": "serif",
                "font.serif": ["Times New Roman"], 
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "lines.linewidth": 2.0,      
                "lines.markersize": 7,       
                "axes.grid": True,
                "grid.alpha": 0.3,           
                "grid.linestyle": "--",      
            })

            style_map = {
                'text':           {'color': '#D2ECBD', 'marker': 'o'}, 
                'image':          {'color': '#FFF2BE', 'marker': 's'}, 
                'audio':          {'color': '#81B0CC', 'marker': 'X'}, 
                'video':          {'color': '#EDA686', 'marker': 'P'}, 
                'text+image':     {'color': '#9467BD', 'marker': 'D'}, 
                'text+audio':     {'color': '#8C564B', 'marker': '^'}, 
                'text+video':     {'color': '#E377C2', 'marker': 'v'}, 
                'text+image_mm':  {'color': '#000000', 'marker': '*'}, 
                'image_mm':       {'color': '#7F7F7F', 'marker': '<'}  
            }

            plt.figure(figsize=(10, 6)) 
            
            modalities = list(aggregated_results.keys())
            if not modalities:
                print("No modalities to plot.")
                return 

            all_layers = sorted([int(k) for k in aggregated_results[modalities[0]].keys()])
            
            sorted_modalities = sorted(modalities, key=lambda x: 1 if "mm" in x else 0)

            # Calculation for Grand Averages
            group_a_mods = {'audio', 'image', 'video', 'text+image', 'text+audio', 'text+video'}
            group_b_mods = {'text+image_mm', 'image_mm'}
            
            group_a_values = []
            group_b_values = []
            
            for modality in sorted_modalities:
                means = [aggregated_results[modality][l]["mean_cosine"] for l in all_layers]
                
                # Collect for grand average
                if modality in group_a_mods:
                     group_a_values.extend(means)
                if modality in group_b_mods:
                     group_b_values.extend(means)
                
                style = style_map.get(modality, {'color': 'black', 'marker': '.'})
                
                if modality in ["text+image_mm", "image_mm"]:
                    ls = '--'
                    lw = 2.5 
                    alpha = 0.9
                    zorder = 10 
                else:
                    ls = '-'
                    lw = 2.0
                    alpha = 0.8
                    zorder = 5

                plt.plot(all_layers, means, 
                        label=modality, 
                        marker=style['marker'], 
                        color=style['color'],
                        linestyle=ls,
                        linewidth=lw,
                        alpha=alpha,
                        zorder=zorder)
            
            # Plot Grand Averages
            if group_a_values:
                avg_a = np.mean(group_a_values)
                plt.axhline(y=avg_a, color='#2CA02C', linestyle='-', linewidth=2.5, alpha=0.7, label='Standard Mean')
            if group_b_values:
                avg_b = np.mean(group_b_values)
                plt.axhline(y=avg_b, color='#D62728', linestyle='--', linewidth=2.5, alpha=0.9, label='Weak Mean')

            plt.title("Semantic Consistency w.r.t Text Modality", fontweight='bold')
            plt.xlabel("Layer Index")
            plt.ylabel("Cosine Similarity")
            
            plt.ylim(0.4, 1.05)
            plt.xlim(min(all_layers)-0.5, max(all_layers)+0.5)

            plt.legend(
                loc='lower left',      
                frameon=True,          
                fancybox=False,        
                edgecolor='black',     
                framealpha=0.9,        
                fontsize=10,
                ncol=2                 
            )
            
            plt.tight_layout()
            
            plot_path = os.path.join(args.output_dir, "semantic_consistency_plot_styled.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Styled plot saved to {plot_path}")

    # Optional: t-SNE visualization per selected/all layers
    if args.do_tsne:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_theme(style="whitegrid", font="Arial", rc={
                "axes.facecolor": "#FFFFFF", 
                "figure.facecolor": "#FFFFFF",
                "grid.color": "#E0E0E0",     
                "grid.linestyle": "--",      
                "axes.edgecolor": "#CCCCCC", 
                "xtick.color": "#555555",    
                "ytick.color": "#555555",
                "text.color": "#333333",     
            })


            custom_palette = {
                'text':           '#D2ECBD', 
                'image':          '#FFF2BE', 
                'audio':          '#81B0CC', 
                'video':          '#EDA686', 
                'text+image':     '#9467BD', 
                'text+audio':     '#8C564B', 
                'text+video':     '#E377C2', 
                'text+image_mm':  '#000000', 
                'image_mm':       '#7F7F7F', 
                'default':        '#95A5A6' 
            }

            marker_map = {
                'text': 'o', 'image': 's', 'audio': 'X', 'video': 'P',
                'text+image': 'D', 'text+audio': '^', 'text+video': 'v',
                'text+image_mm': '*', 'image_mm': '<'
            }

            layers_for_tsne = sorted(tsne_bank.keys()) if args.tsne_layers_all else [l for l in args.tsne_layers if l in tsne_bank]

            for layer in layers_for_tsne:
                X = []
                labels = []
                for modality, vecs in tsne_bank[layer].items():
                    X.extend(vecs)
                    labels.extend([modality] * len(vecs))

                if len(X) < 2:
                    continue

                X = np.stack(X)
                tsne = TSNE(n_components=2, perplexity=min(40, len(X)-1), init="pca", learning_rate="auto", random_state=0)
                X_2d = tsne.fit_transform(X)

                plt.figure(figsize=(10, 8))

                # Scatter Plot
                sns.scatterplot(
                    x=X_2d[:, 0],
                    y=X_2d[:, 1],
                    hue=labels,
                    style=labels,
                    palette=custom_palette, 
                    markers=marker_map,     
                    s=85,                   
                    alpha=0.9,              
                    edgecolor='white',      
                    linewidth=0.8           
                )
                
                # --- NEW: SVM Separator for Weak Modalities ---
                # Define weak classes
                weak_classes = ['text+image_mm', 'image_mm']
                
                # Create binary labels
                y_binary = np.array([1 if lbl in weak_classes else 0 for lbl in labels])
                
                # Only if we have both classes
                if len(np.unique(y_binary)) > 1:
                    clf = LinearSVC(random_state=42, max_iter=10000, dual="auto")
                    clf.fit(X_2d, y_binary)
                    
                    # Get coefficients w1*x + w2*y + b = 0 => y = -(w1*x + b)/w2
                    w = clf.coef_[0]
                    b = clf.intercept_[0]
                    
                    # Plot the line within the current axes limits
                    ax = plt.gca()
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    
                    # Generate x points
                    x_line = np.linspace(x_min, x_max, 100)
                    if abs(w[1]) > 1e-5: # Avoid div by zero
                        y_line = -(w[0] * x_line + b) / w[1]
                        
                        # Filter points to stay within y limits for cleaner plot
                        mask = (y_line >= y_min) & (y_line <= y_max)
                        if np.any(mask):
                             plt.plot(x_line[mask], y_line[mask], color='#D62728', linestyle='--', linewidth=2.0, label='Linear Separator')
                # -----------------------------------------------

                plt.title(f"t-SNE Semantic Space @ Layer {layer}\n(Last-Token States)", fontsize=14, fontweight='bold', pad=15)
                plt.xlabel("t-SNE Dimension 1", fontsize=11)
                plt.ylabel("t-SNE Dimension 2", fontsize=11)

                plt.legend(
                    bbox_to_anchor=(1.02, 1), 
                    loc='upper left', 
                    borderaxespad=0., 
                    frameon=True, 
                    facecolor='white', 
                    edgecolor='#E0E0E0',
                    title="Modality Subsets",
                    title_fontsize=11,
                    fontsize=10
                )

                plt.tight_layout() 
                
                tsne_path = os.path.join(args.output_dir, f"tsne_layer_{layer}_styled.png")
                plt.savefig(tsne_path, dpi=300)
                plt.close()
                print(f"Styled t-SNE plot saved to {tsne_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/consistency_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None, help="Max unique IDs to process")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--do_tsne", action="store_true", help="Run t-SNE on a specific layer")
    parser.add_argument("--tsne_layers", type=str, default="all", help="Comma-separated layer indices for t-SNE; use 'all' to plot every layer")
    parser.add_argument("--tsne_samples", type=int, default=200, help="Max samples per modality per layer for t-SNE")
    parser.add_argument("--force_recompute", action="store_true", help="Ignore cache and recompute everything")
    
    args = parser.parse_args()
    
    # Parse tsne layers
    if args.tsne_layers.strip().lower() == "all":
        args.tsne_layers_all = True
        args.tsne_layers = []
    else:
        args.tsne_layers_all = False
        args.tsne_layers = [int(x) for x in args.tsne_layers.split(',') if x.strip()]

    analyze_similarity(args)
