export CUDA_VISIBLE_DEVICES=1


# python experiment_1_alignment.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file data/alignment_data_adv_seg_t2i_new.json \
#     --output_file output/exp1_results_adv_seg_t2i_new.json

# python experiment_1_alignment.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file data/alignment_data_adv_seg_t2i_new.json \
#     --output_file output/exp1_results_adv_seg_t2i_new.json \
#     # --without_text


# # Section 4.1的实验
# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_t.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_i.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_a.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_v.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_t2i.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_t2a.json \
#     --output_dir output/mechanisms_test

# python experiment_mechanisms.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_t2v.json \
#     --output_dir output/mechanisms_test


# # Section 3.2的实验
# python analyze_semantic_consistency.py \
#   --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#   --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_all_withmm.json \
#   --output_dir output/consistency_analysis \
#   --plot \
#   --max_samples 520 \
#   --do_tsne \
#   --tsne_samples 200

# # Section 4.2的实验
# python experiment_vector_similarity.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign.json \
#     --output_dir output/vector_similarity

# Section 4.3和5.1的实验
python exp_vector_analysis.py \
    --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
    --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign_final.json \
    --output_dir output/vector_similarity_draw \
    # --pca_only_target_modalities

# python residual_analysis.py \
#   --output_dir output/vector_similarity \
#   --layers 15,16,17,18,19,20,21,22,23,24 \
#   --save_dir output/residual_analysis \
#   --plot

# python analyze_projector_norms.py \
#   --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#   --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign_final.json \
#   --layers 18,19,20 \
#   --max_per_modality 64 \
#   --output output/projector_norms.json \
#   --device cuda


# python exp_vector_analysis_baichuan.py \
#   --model_path /mnt/data/miyan/.cache/modelscope/hub/models/baichuan-inc/Baichuan-Omni-1d5 \
#   --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign_final.json \
#   --output_dir output/vector_similarity_baichuan 


# python exp_vector_analysis_minicpm_o.py \
#   --model_path /mnt/data/miyan/.cache/modelscope/hub/models/OpenBMB/MiniCPM-o-2_6 \
#   --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign_final.json \
#   --output_dir output/vector_similarity_minicpm_o \


# python project_text_vector.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/Qwen/Qwen2.5-Omni-7B \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_all_with_benign_final.json \
#     --canonical_dir output/vector_similarity \
#     --output_file output/text_projection_results.json


# python experiment_mechanisms_baichuan.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/baichuan-inc/Baichuan-Omni-1d5 \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_t2v.json \
#     --output_dir output/mechanisms_test \


# python experiment_mechanisms_minicpm_o.py \
#     --model_path /mnt/data/miyan/.cache/modelscope/hub/models/OpenBMB/MiniCPM-o-2_6 \
#     --data_file /mnt/data/miyan/user_simulation/omni_safety_research/data/alignment_data_adv_seg_t2v.json \
#     --output_dir output/mechanisms_test \