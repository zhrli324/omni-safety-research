import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils import load_model
from qwen_omni_utils import process_mm_info

class LayerInputNormHook:
    def __init__(self, layer_module):
        self.handle = layer_module.register_forward_pre_hook(self._hook)
        self.records = []

    def _hook(self, module, inputs):
        if not inputs:
            return
        hidden = inputs[0]
        if hidden is None:
            return
        with torch.no_grad():
            # Cast to float32 to avoid downstream numpy dtype issues (e.g., bfloat16)
            norm = torch.norm(hidden, dim=-1).float()  # [batch, seq]
            self.records.append(norm.detach().cpu())

    def clear(self):
        self.records = []

    def remove(self):
        self.handle.remove()


def prepare_inputs(processor, item, model):
    modality = item.get("modality", "text")
    use_audio_in_video = False
    content = []

    if "image" in modality and item.get("image_path"):
        content.append({"type": "image", "image": item["image_path"]})
    if "video" in modality and item.get("video_path"):
        content.append({"type": "video", "video": item["video_path"]})
    if "audio" in modality and item.get("audio_path"):
        content.append({"type": "audio", "audio": item["audio_path"]})
    if "text" in modality and item.get("text_instruction"):
        content.append({"type": "text", "text": item["text_instruction"]})
    elif "text" in modality and item.get("text"):
        content.append({"type": "text", "text": item["text"]})

    conversation = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                }
            ],
        },
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    return inputs


def collect_norms(model, processor, data, layers_to_probe, max_per_modality=64):
    # layers_to_probe: list of layer indices to probe pre-forward hidden
    layers = None
    if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "layers"):
        layers = model.thinker.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError("Cannot locate model layers for hooks.")

    hooks = {idx: LayerInputNormHook(layers[idx]) for idx in layers_to_probe}

    per_modality_norms = {}
    seen = {m: 0 for m in ["text", "image", "audio", "video", "text+image", "text+audio", "text+video"]}

    for item in tqdm(data, desc="Collecting norms"):
        modality = item.get("modality", "text")
        if modality not in seen:
            continue
        if seen[modality] >= max_per_modality:
            continue

        try:
            inputs = prepare_inputs(processor, item, model)
        except Exception:
            continue

        hooks_records_before = {k: len(h.records) for k, h in hooks.items()}
        with torch.no_grad():
            model.thinker(**inputs)
        seen[modality] += 1

        for layer_idx, hook in hooks.items():
            if len(hook.records) == 0:
                continue
            if len(hook.records) == hooks_records_before[layer_idx]:
                continue
            norms = hook.records[-1]  # [batch, seq]
            flat = norms.flatten().numpy()
            stats = {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "p50": float(np.percentile(flat, 50)),
                "p90": float(np.percentile(flat, 90)),
                "p99": float(np.percentile(flat, 99)),
            }
            per_modality_norms.setdefault(modality, {}).setdefault(layer_idx, []).append(stats)

    # Aggregate stats per modality per layer
    aggregated = {}
    for modality, layer_dict in per_modality_norms.items():
        aggregated[modality] = {}
        for layer_idx, stats_list in layer_dict.items():
            if not stats_list:
                continue
            mean_vals = {k: np.mean([s[k] for s in stats_list]) for k in stats_list[0]}
            aggregated[modality][layer_idx] = mean_vals

    for h in hooks.values():
        h.remove()

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Compare token norm magnitude across modalities to probe representation compression.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True, help="JSON file containing text_train and cross_modal_test fields.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default="15", help="Comma-separated layer indices to probe (pre-forward hook).")
    parser.add_argument("--max_per_modality", type=int, default=64)
    parser.add_argument("--output", type=str, default="output/projector_norms.json")
    args = parser.parse_args()

    model, processor = load_model(args.model_path, args.device)

    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # merge subsets for sampling
    merged = []
    for key in ["text_train", "cross_modal_test", "train", "test"]:
        if key in data and isinstance(data[key], list):
            merged.extend(data[key])

    if not merged:
        raise ValueError("No data found in provided file.")

    layers_to_probe = [int(x) for x in args.layers.split(",") if x.strip()]

    aggregated = collect_norms(model, processor, merged, layers_to_probe, max_per_modality=args.max_per_modality)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved norm stats to {args.output}")

    # Pretty print summary
    for modality, layer_dict in aggregated.items():
        for layer_idx, stats in layer_dict.items():
            print(f"Modality {modality} | Layer {layer_idx} | mean={stats['mean']:.3f} std={stats['std']:.3f} p50={stats['p50']:.3f} p90={stats['p90']:.3f} p99={stats['p99']:.3f}")


if __name__ == "__main__":
    main()
