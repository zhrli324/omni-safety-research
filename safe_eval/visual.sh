
python visualize_hidden_states.py \
    --baseline qwen/omnibench_run1_tsne_hidden_states.pt \
    --adapter qwen/sr_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_sr_qwen.png

python visualize_hidden_states.py \
    --baseline qwen/omnibench_run1_tsne_hidden_states.pt \
    --adapter qwen/ada_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_ada_qwen.png

python visualize_hidden_states.py \
    --baseline baichuan/omnibench_run1_tsne_hidden_states.pt \
    --adapter baichuan/ada_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_ada_baichuan.png

python visualize_hidden_states.py \
    --baseline baichuan/omnibench_run1_tsne_hidden_states.pt \
    --adapter baichuan/sr_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_sr_baichuan.png

python visualize_hidden_states.py \
    --baseline minicpm_o/omnibench_run1_tsne_hidden_states.pt \
    --adapter minicpm_o/ada_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_ada_minicpm_o.png

python visualize_hidden_states.py \
    --baseline minicpm_o/omnibench_run1_tsne_hidden_states.pt \
    --adapter minicpm_o/sr_omnibench_run1_tsne_hidden_states.pt \
    --label_base "Original Model" \
    --label_adapt "With Safety Method" \
    --output comparison_tsne_sr_minicpm_o.png
    
# qwen/og_omnibench_run1_tsne_hidden_states.pt
# qwen/sr_omnibench_run1_tsne_hidden_states.pt

# baichuan/ada_omnibench_run1_tsne_hidden_states.pt