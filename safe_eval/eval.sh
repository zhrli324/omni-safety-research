#!/usr/bin/env bash
set -u

script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir" || exit 1

datasets_map=(
    # "benign_t2i:benign_t2i"
    # "benign_t2v:benign_t2v"
    # "advbench_t2v:advbench_t2v"
    # "advbench_t2i:advbench_t2i"
    # "advbench_t2a:advbench_t2a"
    # "harmbench:harmbench"
    "beavertail_test:beavertail_test"
    # "mmsafetybench:mmsafetybench"
    # "holisafe:holisafe"
    # "video_safetybench:video_safetybench"
    # "omnisafetybench_iat:omnisafetybench_iat"
    # "omnisafetybench_vat:omnisafetybench_vat"
    # "harmbench_audio:harmbench_audio"
    # "beavertail_audio_1k:beavertail_audio_1k"
    # "pair_audio:pair_audio"
    # "omnibench:omnibench"
)

# prefixes=("" "rs_" "sr_")
# prefixes=("" "rs_" "sr_" "og_" "svg_")
prefixes=("" "sr_" "ada_" "og_")
# prefixes=("og_")

# 定义运行次数
runs=(1)



OUTDIR=""

echo "Starting Batch Evaluation..."
echo "Output Directory: $OUTDIR"

for entry in "${datasets_map[@]}"; do
    filename_part="${entry%%:*}"
    dataset_arg="${entry##*:}"

    for prefix in "${prefixes[@]}"; do
        for run in "${runs[@]}"; do
            filename="${prefix}${filename_part}_run${run}.jsonl"
            filepath="$OUTDIR/$filename"
            
            if [[ -f "$filepath" ]]; then
                echo "----------------------------------------------------------------"
                echo "Evaluating File: $filename"
                echo "  - Method Prefix : '${prefix:-vanilla}'"
                echo "  - Dataset Arg   : $dataset_arg"
                echo "  - Run Number    : $run"
                
                stats_path="${filepath%.jsonl}_stats.json"
                
                if python evaluation.py \
                    --input_path "$filepath" \
                    --output_path "$stats_path" \
                    --dataset "$dataset_arg" \
                    --num_workers 5 ; then
                    echo "Status: Success"
                else
                    echo "Status: Failed"
                fi
            # else
                # echo "Skipping missing file: $filename"
            fi
        done
    done
done

echo "----------------------------------------------------------------"
echo "Batch Evaluation Complete."


