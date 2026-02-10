import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import seaborn as sns

from utils import load_model, HookManager, calculate_refusal_vector
from qwen_omni_utils import process_mm_info

# ==========================================
# ==========================================
PLOT_CONFIG = {
    'font_family': 'serif',
    'font_name': 'Times New Roman',
    'figure_size': (9, 9),
    'dpi': 300,
    'transparent_save': True,

    'font_size_title': 20,
    'font_size_label': 14,
    'font_size_tick': 12,
    'font_size_annot': 13,

    'styles': {
        'text':       {'color': "#3BF13B", 'marker': 'o'}, # Blue
        'image':      {'color': "#EFC728", 'marker': 's'}, # Orange
        'audio':      {'color': "#3FACEA", 'marker': '^'}, # Green
        'video':      {'color': '#EDA686', 'marker': 'D'}, # Red
        'text+image': {'color': "#963EE9", 'marker': 'P'}, # Purple
        'text+audio': {'color': '#8C564B', 'marker': 'X'}, # Brown
        'text+video': {'color': "#F84EC5", 'marker': '*'}, # Pink
    },

    'default_style': {'color': 'gray', 'marker': '.'},

    'arrow_width': 0.03,
    'arrow_alpha': 1.0,
    'point_size': 130,
    
    'circle_color': '#B0B0B0',
    'circle_alpha': 1.0,
    'circle_linestyle': '--',
    
    'crosshair_color': '#B0B0B0',
    'crosshair_alpha': 0.2,
}

plt.rcParams['font.family'] = PLOT_CONFIG['font_family']
plt.rcParams['font.serif'] = [PLOT_CONFIG['font_name']]
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# ==========================================
def draw_vector_pca(ax, vectors_2d, labels, norms, title, xlabel, ylabel):
    """
    vectors_2d: (N, 2) numpy array
    labels: list of modality names
    norms: list of original vector norms (for annotation)
    """
    max_radius = np.max(np.linalg.norm(vectors_2d, axis=1)) * 1.15
    
    intervals = [0.25, 0.50, 0.75, 1.00]
    for r_factor in intervals:
        r = max_radius * r_factor
        circle = patches.Circle((0, 0), r, fill=False, 
                                color=PLOT_CONFIG['circle_color'], 
                                linestyle=PLOT_CONFIG['circle_linestyle'], 
                                alpha=PLOT_CONFIG['circle_alpha'])
        ax.add_patch(circle)
        # ax.text(r, 0, f'{r:.1f}', fontsize=8, color='gray', ha='center', va='bottom')

    # [Configuration/Note]
    ax.axhline(0, color=PLOT_CONFIG['crosshair_color'], linestyle='-', alpha=PLOT_CONFIG['crosshair_alpha'], zorder=0)
    ax.axvline(0, color=PLOT_CONFIG['crosshair_color'], linestyle='-', alpha=PLOT_CONFIG['crosshair_alpha'], zorder=0)

    for i, label in enumerate(labels):
        x, y = vectors_2d[i]
        
        style = PLOT_CONFIG['styles'].get(label, PLOT_CONFIG['default_style'])
        color = style['color']
        marker = style['marker']
        
        ax.arrow(0, 0, x, y, 
                 color=color, alpha=PLOT_CONFIG['arrow_alpha'], 
                 width=PLOT_CONFIG['arrow_width'], 
                 length_includes_head=True, zorder=2)
        
        ax.scatter(x, y, s=PLOT_CONFIG['point_size'], 
                   color=color, marker=marker, 
                   edgecolor='white', linewidth=1.5,
                   label=label, zorder=3)
        
        # offset_x = 0.5 if x >= 0 else -0.5
        # offset_y = 0.5 if y >= 0 else -0.5
        
        # [Configuration/Note]
        
        # ax.annotate(text_label, (x, y), 
        #             xytext=(10 * offset_x, 10 * offset_y), textcoords='offset points',
        #             fontsize=PLOT_CONFIG['font_size_annot'], 
        #             color='black',
        #             fontname=PLOT_CONFIG['font_name'],
        #             ha='left' if x > 0 else 'right',
        #             va='bottom' if y > 0 else 'top',
        #             zorder=4,
        # [Configuration/Note]

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal')
    
    ax.set_title(title, fontsize=PLOT_CONFIG['font_size_title'], fontname=PLOT_CONFIG['font_name'], pad=15, weight='bold')
    ax.set_xlabel(xlabel, fontsize=PLOT_CONFIG['font_size_label'], fontname=PLOT_CONFIG['font_name'])
    ax.set_ylabel(ylabel, fontsize=PLOT_CONFIG['font_size_label'], fontname=PLOT_CONFIG['font_name'])
    
    ax.tick_params(axis='both', which='major', labelsize=PLOT_CONFIG['font_size_tick'])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
        
    # ax.legend(loc='upper right', frameon=True, framealpha=1, fontsize=10)


# ==========================================
# ==========================================
def run_vector_analysis(args):
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)
    
    hook_manager = HookManager(model)
    target_layers = list(range(15, 25))
    layer_path = "thinker.model.layers" 
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    vectors = {layer: {} for layer in target_layers}

    cache_dir = args.output_dir
    os.makedirs(cache_dir, exist_ok=True)
    cache_paths = {layer: os.path.join(cache_dir, f"refusal_vectors_layer_{layer}.pt") for layer in target_layers}
    cached_layers = []
    for layer, path in cache_paths.items():
        if os.path.exists(path):
            try:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, dict):
                    vectors[layer] = loaded
                    cached_layers.append(layer)
            except Exception:
                pass

    compute_needed = len(cached_layers) < len(target_layers)
    
    # --- Step 1: Calculate Text Vectors ---
    if compute_needed:
        print("Calculating Text Refusal Vectors...")
        text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
        train_subset = data["text_train"]
        
        for item in tqdm(train_subset, desc="Text Data"):
            content = [{"type": "text", "text": item["text"]}]
            conversation = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}, {"role": "user", "content": content}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, True) # Dummy True for speed? or False
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model.thinker(**inputs)
                
            for layer in target_layers:
                act = hook_manager.activations[layer][:, -1, :].cpu()
                text_acts[layer][item["type"]].append(act)
            hook_manager.activations = {}

        for layer in target_layers:
            if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
                vectors[layer]["text"] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    # --- Step 2: Calculate MM Vectors ---
    # target_modalities = ["image", "audio", "video", "text+image", "text+audio", "text+video"]
    target_modalities = ["text", "image", "audio", "video"]
    mm_acts = {m: {layer: {"safe": [], "harmful": []} for layer in target_layers} for m in target_modalities}
    max_samples = 200
    counts = {m: {"safe": 0, "harmful": 0} for m in target_modalities}
    
    if compute_needed:
        print("Calculating Multi-Modal Refusal Vectors...")
        for item in tqdm(data["cross_modal_test"], desc="MM Data"):
            modality = item.get("modality", "")
            label = item.get("label", "harmful")
            
            if modality not in target_modalities: continue
            if counts[modality][label] >= max_samples: continue
            
            content = []
            if "image" in modality: content.append({"type": "image", "image": item["image_path"]})
            if "audio" in modality: content.append({"type": "audio", "audio": item["audio_path"]})
            if "video" in modality: content.append({"type": "video", "video": item["video_path"]})
            if "text+" in modality: content.append({"type": "text", "text": item["text_instruction"]})
            
            conversation = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}, {"role": "user", "content": content}]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model.thinker(**inputs)
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

        for layer in target_layers:
            if vectors[layer]:
                torch.save(vectors[layer], cache_paths[layer])

    print("Analyzing and Plotting...")
    os.makedirs(args.output_dir, exist_ok=True)

    analysis_results = {}
    
    for layer in target_layers:
        layer_vecs = vectors[layer]
        if len(layer_vecs) < 2: continue

        base_vec = layer_vecs.get("text")
        aligned_vecs = []
        modality_names = list(layer_vecs.keys())
        if args.pca_only_target_modalities:
            modality_names = [m for m in target_modalities if m in layer_vecs]
            if len(modality_names) < 2:
                continue

        if base_vec is not None:
            base_norm = torch.norm(base_vec) + 1e-6
            for m in modality_names:
                v = layer_vecs[m]
                cos_sim = torch.dot(v.to(base_vec.dtype), base_vec) / (torch.norm(v) * base_norm)
                v_aligned = v if cos_sim >= 0 else -v
                aligned_vecs.append(v_aligned)
        else:
            aligned_vecs = [layer_vecs[m] for m in modality_names]

        X_raw = torch.stack(aligned_vecs).float().numpy()
        norms_raw = [float(np.linalg.norm(v)) for v in X_raw]

        pca_dim = min(len(modality_names), 2)
        
        # 1. Raw PCA
        pca_raw = PCA(n_components=pca_dim, random_state=0)
        pca_raw.fit(X_raw)
        X_pca_raw = pca_raw.transform(X_raw)
        exp_var_raw = pca_raw.explained_variance_ratio_

        # 2. Norm PCA
        X_unit = X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-12)
        pca_norm = PCA(n_components=pca_dim, random_state=0)
        pca_norm.fit(X_unit)
        X_pca_norm = pca_norm.transform(X_unit)
        exp_var_norm = pca_norm.explained_variance_ratio_

        if pca_raw.n_components_ >= 2:
            fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
            draw_vector_pca(
                ax, 
                X_pca_raw[:, :2],
                modality_names, 
                norms_raw,
                title=f"Layer {layer} Refusal Vectors PCA (Raw)",
                xlabel=f"PC1 ({exp_var_raw[0]:.1%})",
                ylabel=f"PC2 ({exp_var_raw[1]:.1%})"
            )
            save_path = os.path.join(args.output_dir, f"layer_{layer}_pca_raw_beautiful.png")
            plt.savefig(save_path, bbox_inches='tight', transparent=PLOT_CONFIG['transparent_save'], dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)

        if pca_norm.n_components_ >= 2:
            fig, ax = plt.subplots(figsize=PLOT_CONFIG['figure_size'])
            draw_vector_pca(
                ax, 
                X_pca_norm[:, :2],
                modality_names, 
                [1.0]*len(modality_names),
                title=f"Layer {layer} Refusal Vectors PCA (Direction Only)",
                xlabel=f"PC1 ({exp_var_norm[0]:.1%})",
                ylabel=f"PC2 ({exp_var_norm[1]:.1%})"
            )
            save_path = os.path.join(args.output_dir, f"layer_{layer}_pca_norm_beautiful.png")
            plt.savefig(save_path, bbox_inches='tight', transparent=PLOT_CONFIG['transparent_save'], dpi=PLOT_CONFIG['dpi'])
            plt.close(fig)

        plt.figure(figsize=(10, 5))
        sns.set_theme(style="whitegrid")
        bar_colors = [PLOT_CONFIG['styles'].get(m, PLOT_CONFIG['default_style'])['color'] for m in modality_names]
        ax = sns.barplot(x=modality_names, y=norms_raw, palette=bar_colors)
        ax.set_title(f"Layer {layer} Refusal Vector Norms", fontsize=18, fontname="Times New Roman", pad=15)
        ax.set_ylabel("L2 Norm", fontsize=14, fontname="Times New Roman")
        plt.xticks(rotation=45, fontsize=12, fontname="Times New Roman")
        plt.yticks(fontsize=12, fontname="Times New Roman")
        sns.despine(left=False, bottom=False, right=False, top=False)
        save_path = os.path.join(args.output_dir, f"layer_{layer}_norms_beautiful.png")
        plt.savefig(save_path, bbox_inches='tight', transparent=PLOT_CONFIG['transparent_save'], dpi=PLOT_CONFIG['dpi'])
        plt.close()

    print(f"Analysis done. Beautiful results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/vector_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pca_only_target_modalities",
        action="store_true",
        help="PCA 仅对 target_modalities 中的模态进行降维可视化"
    )
    args = parser.parse_args()
    run_vector_analysis(args)
