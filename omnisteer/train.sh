export CUDA_VISIBLE_DEVICES=1


# python train_safety_adapter_qwen.py \
#     --model_path Qwen/Qwen2.5-Omni-7B \
#     --data_file data/train.jsonl \
#     --refusal_vector_dir omni_safety_research/output/SVD_qwen \
#     --target_layer "15,16,17" \
#     --output_dir omni_safety_research/output/adapter_qwen \
#     --epochs 3 \


# python train_safety_adapter_baichuan.py \
#     --model_path baichuan-inc/Baichuan-Omni-1d5 \
#     --data_file data/train.jsonl \
#     --refusal_vector_dir omni_safety_research/output/SVD_baichuan \
#     --target_layer "15,16,17,18,19,20" \
#     --output_dir omni_safety_research/output/adapter_baichuan \
#     --epochs 3

python train_safety_adapter_minicpm_o.py \
    --model_path OpenBMB/MiniCPM-o-2_6 \
    --data_file data/train.jsonl \
    --refusal_vector_dir omni_safety_research/output/SVD_minicpm_o \
    --target_layer "13,14,15,16,17" \
    --output_dir omni_safety_research/output/adapter_minicpm_o \
    --epochs 3