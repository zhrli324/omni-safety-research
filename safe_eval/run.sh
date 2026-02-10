#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1,3,4,5,6

inputs=(
    # "data/harmbench.jsonl"
    # "data/beavertail_test.jsonl"
    # "data/mmsafetybench.jsonl"
    # "data/holisafe.jsonl"
    # "data/advbench_i.jsonl"
    # "data/advbench_t.jsonl"
    # "data/advbench_a.jsonl"
    # "data/advbench_v.jsonl"
    # "data/advbench_t2a.jsonl"
    # "data/advbench_t2v.jsonl"
    # "data/advbench_t2i.jsonl"
    # "data/advbench_t2ia.jsonl"
    # "data/advbench_t2va.jsonl"
    # "data/advbench_a2i.jsonl"
    # "data/advbench_a2v.jsonl"
    # "data/omnisafetybench_iat.jsonl"
    # "data/omnisafetybench_vat.jsonl"
    # "data/harmbench_audio.jsonl"
    # "data/pair_audio.jsonl"
    "data/OmniBench/omnibench"
)
outdir="output/qwen"
model="Qwen/Qwen2.5-Omni-7B"
data_dir="EOSA"

for run in 1; do
    for infile in "${inputs[@]}"; do
        base=$(basename "$infile" .jsonl)
        outfile="${outdir}/${base}_run${run}_tsne.jsonl"
        python unified_eval_qwen.py \
            --input_file "$infile" \
            --output_file "$outfile" \
            --data_dir "$data_dir" \
            --model_path "$model" \
            --save_hidden_states
            # --adapter_path omni_safety_research/output/adapter_qwen/safety_adapter_epoch_0.pt \
            # --adapter_layers "15,16,17,18,19,20,21,22,23,24" \
            # --adapter_vector_dir omni_safety_research/output/SVD_qwen
    done
done

for run in 1; do
    for infile in "${inputs[@]}"; do
        base=$(basename "$infile" .jsonl)
        outfile="${outdir}/ada_${base}_run${run}_tsne.jsonl"
        python unified_eval_qwen.py \
            --input_file "$infile" \
            --output_file "$outfile" \
            --data_dir "$data_dir" \
            --model_path "$model" \
            --adapter_path omni_safety_research/output/adapter_qwen/safety_adapter_step_300.pt \
            --adapter_layers "15,16,17" \
            --adapter_vector_dir omni_safety_research/output/SVD_qwen \
            --save_hidden_states
    done
done

# for run in 1; do
#     for infile in "${inputs[@]}"; do
#         base=$(basename "$infile" .jsonl)
#         outfile="${outdir}/og_${base}_run${run}_tsne.jsonl"
#         python unified_eval_qwen.py \
#             --input_file "$infile" \
#             --output_file "$outfile" \
#             --data_dir "$data_dir" \
#             --model_path "$model" \
#             --omniguard_path "../OMNIGUARD/omniguard_classifier_qwen.pt" \
#             --omniguard_layer 27 \
#             --save_hidden_states
#     done
# done

for run in 1; do
    for infile in "${inputs[@]}"; do
        base=$(basename "$infile" .jsonl)
        outfile="${outdir}/sr_${base}_run${run}_tsne.jsonl"
        python unified_eval_qwen.py \
            --input_file "$infile" \
            --output_file "$outfile" \
            --data_dir "$data_dir" \
            --model_path "$model" \
            --use_self_reminder \
            --save_hidden_states
    done
done