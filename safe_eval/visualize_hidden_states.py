import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.manifold import TSNE
import argparse
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd

# ==========================================
# ==========================================
CONFIG = {
    'font_family': 'serif',
    'font_name': 'Times New Roman',
    'font_size_title': 20,
    'font_size_label': 15,
    'font_size_tick': 12,
    
    'figure_width': 12,
    'figure_height': 8,
    
    'color_palette': ["#84D9EF", "#D37E7D"],
    
    'point_size': 18,
    'point_alpha': 0.9,
    'point_edge_width': 0,
    
    'kde_fill': True,
    'kde_alpha': 0.3,
    
    'ellipse_std': 2.0,           
    'ellipse_fill_alpha': 0.1,    
    'ellipse_edge_alpha': 0.8,   
    'ellipse_ls': '--',
    'ellipse_lw': 1.5,
    
    'grid_on': True,              
    'grid_alpha': 0.2,
    'grid_style': '--',           
}

plt.rcParams['font.family'] = CONFIG['font_family']
plt.rcParams['font.serif'] = [CONFIG['font_name']]

# ==========================================
# ==========================================
def load_hidden_states(file_path):
    print(f"Loading data from {file_path}...")
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.random.randn(500, 768), list(range(500))

    vectors = []
    ids = []
    sorted_keys = sorted(data.keys(), key=lambda x: int(x) if isinstance(x, int) or (isinstance(x, str) and x.isdigit()) else str(x))
    
    for k in sorted_keys:
        v = data[k]
        if isinstance(v, torch.Tensor):
            vectors.append(v.float().numpy())
        else:
            vectors.append(np.array(v))
        ids.append(k)
        
    vectors = np.array(vectors)
    if len(vectors.shape) > 2:
        vectors = vectors.reshape(vectors.shape[0], -1)
    elif len(vectors.shape) == 1:
        vectors = vectors.reshape(vectors.shape[0], -1)
        
    return vectors, ids

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', edgecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, edgecolor=edgecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# ==========================================
# ==========================================

def main(file1, file2, label1, label2, output_file, perplexity=30):
    data1, ids1 = load_hidden_states(file1)
    data2, ids2 = load_hidden_states(file2)
    combined_data = np.concatenate([data1, data2], axis=0)
    print(f"Running t-SNE on {combined_data.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(combined_data)
    tsne1 = tsne_results[:len(data1)]
    tsne2 = tsne_results[len(data1):]
    df1 = pd.DataFrame(tsne1, columns=['x', 'y'])
    df1['Label'] = label1
    df2 = pd.DataFrame(tsne2, columns=['x', 'y'])
    df2['Label'] = label2
    df = pd.concat([df1, df2])

    print("Generating clean plot material...")
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    palette = {label1: CONFIG['color_palette'][0], label2: CONFIG['color_palette'][1]}

    g = sns.JointGrid(data=df, x="x", y="y", hue="Label", palette=palette, 
                      height=CONFIG['figure_height'], ratio=6, space=0.1)
    g.fig.set_size_inches(CONFIG['figure_width'], CONFIG['figure_height'])

    g.plot_marginals(sns.kdeplot, fill=CONFIG['kde_fill'], alpha=CONFIG['kde_alpha'], linewidth=1.5, common_norm=False)
    
    g.ax_marg_x.axis('off')
    g.ax_marg_y.axis('off')

    g.plot_joint(sns.scatterplot, 
                 s=CONFIG['point_size'], 
                 alpha=CONFIG['point_alpha'], 
                 edgecolor="white", 
                 linewidth=CONFIG['point_edge_width'],
                 legend=False,
                 zorder=10) 

    ax = g.ax_joint

    if CONFIG['grid_on']:
        ax.grid(True, linestyle=CONFIG['grid_style'], alpha=CONFIG['grid_alpha'], color='gray', zorder=0)

    color1_rgb = mcolors.to_rgb(palette[label1])
    fill_color1 = (*color1_rgb, CONFIG['ellipse_fill_alpha'])
    edge_color1 = (*color1_rgb, CONFIG['ellipse_edge_alpha'])
    confidence_ellipse(tsne1[:, 0], tsne1[:, 1], ax, n_std=CONFIG['ellipse_std'], 
                       facecolor=fill_color1, edgecolor=edge_color1,
                       linestyle=CONFIG['ellipse_ls'], linewidth=CONFIG['ellipse_lw'], zorder=5)

    color2_rgb = mcolors.to_rgb(palette[label2])
    fill_color2 = (*color2_rgb, CONFIG['ellipse_fill_alpha'])
    edge_color2 = (*color2_rgb, CONFIG['ellipse_edge_alpha'])
    confidence_ellipse(tsne2[:, 0], tsne2[:, 1], ax, n_std=CONFIG['ellipse_std'], 
                       facecolor=fill_color2, edgecolor=edge_color2,
                       linestyle=CONFIG['ellipse_ls'], linewidth=CONFIG['ellipse_lw'], zorder=5)

    # ==========================================
    # ==========================================
    # ax.legend(...) 

    ax.set_xlabel('')
    ax.set_ylabel('')
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    # plt.suptitle(...)
    
    # ==========================================
    # ==========================================
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    print(f"Saving clean figure to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight', dpi=300, transparent=True)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline.pt")
    parser.add_argument("--adapter", default="adapter.pt")
    parser.add_argument("--label_base", default="Original Model")
    parser.add_argument("--label_adapt", default="With Safety Method")
    parser.add_argument("--output", default="clean_tsne_material.png")
    parser.add_argument("--perplexity", type=int, default=30)
    args = parser.parse_args()
    
    if not os.path.exists(args.baseline):
        print("Running in DEMO mode (generating fake data)...")
        
    main(args.baseline, args.adapter, args.label_base, args.label_adapt, args.output, args.perplexity)