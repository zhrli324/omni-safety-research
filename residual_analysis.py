import os
import json
import argparse
import torch
import numpy as np
from itertools import combinations
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def load_vectors(output_dir, layers):
    vectors = {}
    for layer in layers:
        path = os.path.join(output_dir, f"old_refusal_vector/refusal_vectors_layer_{layer}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}, please run exp_vector_analysis.py first.")
        vectors[layer] = torch.load(path, map_location="cpu")
    return vectors


def load_gold(output_dir, layers):
    gold = {}
    for layer in layers:
        path = os.path.join(output_dir, f"canonical_refusal_vector_layer_{layer}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}, please run exp_vector_analysis.py first to get canonical vectors.")
        gold[layer] = torch.load(path, map_location="cpu")
    return gold


def compute_residuals(vectors, gold, layers, target_modalities=None):
    residuals = {}
    stats = {}
    for layer in layers:
        layer_vecs = vectors.get(layer, {})
        gold_vec = gold[layer]
        residuals[layer] = {}
        stats[layer] = {}
        for modality, vec in layer_vecs.items():
            if target_modalities and modality not in target_modalities:
                continue
            r = vec.to(torch.float32) - gold_vec.to(torch.float32)
            residuals[layer][modality] = r.cpu()
            # metrics
            norm_r = torch.norm(r).item()
            norm_v = torch.norm(vec).item()
            norm_g = torch.norm(gold_vec).item()
            cos_rg = torch.dot(r.flatten(), gold_vec.flatten()) / (torch.norm(r) * torch.norm(gold_vec) + 1e-12)
            cos_vg = torch.dot(vec.flatten(), gold_vec.flatten()) / (torch.norm(vec) * torch.norm(gold_vec) + 1e-12)
            stats[layer][modality] = {
                "residual_norm": norm_r,
                "vector_norm": norm_v,
                "gold_norm": norm_g,
                "cos_vec_gold": cos_vg.item() if hasattr(cos_vg, "item") else float(cos_vg),
                "cos_residual_gold": cos_rg.item() if hasattr(cos_rg, "item") else float(cos_rg),
            }
    return residuals, stats


def pairwise_cos(residuals):
    cos = {}
    for layer, mod_dict in residuals.items():
        cos[layer] = {}
        keys = list(mod_dict.keys())
        for m1, m2 in combinations(keys, 2):
            r1 = mod_dict[m1].flatten()
            r2 = mod_dict[m2].flatten()
            val = torch.dot(r1, r2) / (torch.norm(r1) * torch.norm(r2) + 1e-12)
            cos[layer][f"{m1}|{m2}"] = val.item()
    return cos


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_pca(residuals, layer, out_dir):
    mod_names = list(residuals[layer].keys())
    if len(mod_names) < 2:
        return
    X = torch.stack([residuals[layer][m] for m in mod_names]).float().numpy()
    pca = PCA(n_components=min(2, X.shape[0]))
    X2d = pca.fit_transform(X)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X2d[:, 0], y=X2d[:, 1], hue=mod_names, style=mod_names, s=120)
    for i, m in enumerate(mod_names):
        plt.annotate(m, (X2d[i, 0], X2d[i, 1]), fontsize=12)
        plt.arrow(0, 0, X2d[i, 0], X2d[i, 1], color="gray", alpha=0.25, width=0.002)
    evr = pca.explained_variance_ratio_
    plt.xlabel(f"PC1 ({evr[0]:.2%})")
    if len(evr) > 1:
        plt.ylabel(f"PC2 ({evr[1]:.2%})")
    plt.title(f"Residual PCA Layer {layer}")
    plt.grid(True, alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"residual_pca_layer_{layer}.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Residual analysis of modality refusal vectors vs SVD gold vector.")
    parser.add_argument("--output_dir", type=str, default="output/vector_analysis", help="Directory that contains refusal_vectors_layer_*.pt and canonical_refusal_vector_layer_*.pt")
    parser.add_argument("--layers", type=str, default="19", help="Comma-separated layers to analyze")
    parser.add_argument("--modalities", type=str, default="", help="Comma-separated modalities to keep (default: all present)")
    parser.add_argument("--save_dir", type=str, default="output/residual_analysis")
    parser.add_argument("--plot", action="store_true", help="Save PCA plots of residuals per layer")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    target_modalities = [m for m in args.modalities.split(",") if m.strip()] if args.modalities else None

    vectors = load_vectors(args.output_dir, layers)
    gold = load_gold(args.output_dir, layers)

    residuals, stats = compute_residuals(vectors, gold, layers, target_modalities)
    cos = pairwise_cos(residuals)

    os.makedirs(args.save_dir, exist_ok=True)
    save_json(stats, os.path.join(args.save_dir, "residual_stats.json"))
    save_json(cos, os.path.join(args.save_dir, "residual_pairwise_cos.json"))

    # also persist residual tensors for potential steering ablations
    for layer in layers:
        torch.save(residuals[layer], os.path.join(args.save_dir, f"residuals_layer_{layer}.pt"))
        if args.plot:
            plot_pca(residuals, layer, args.save_dir)

    print("Residual analysis done.")
    print(f"Stats saved to {os.path.join(args.save_dir, 'residual_stats.json')}")
    print(f"Pairwise cos saved to {os.path.join(args.save_dir, 'residual_pairwise_cos.json')}")

if __name__ == "__main__":
    main()
