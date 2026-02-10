import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np
import logging
import warnings
from typing import List, Dict, Any
from utils import HookManager, calculate_refusal_vector
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Baichuan-Omni helpers (adapted from safe_eval/unified_eval_baichuan.py)
# -----------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Patch Qwen2VLVisionBlock to handle rotary_pos_emb vs position_embeddings mismatch
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionBlock

    def patch_qwen2_vl_vision_block():
        original_forward = Qwen2VLVisionBlock.forward

        def patched_forward(self, hidden_states, attention_mask=None, position_embeddings=None, rotary_pos_emb=None, **kwargs):
            if position_embeddings is None and rotary_pos_emb is not None:
                if isinstance(rotary_pos_emb, (list, tuple)):
                    if len(rotary_pos_emb) > 2:
                        position_embeddings = rotary_pos_emb[:2]
                    else:
                        position_embeddings = rotary_pos_emb
                elif hasattr(rotary_pos_emb, "shape"):
                    if rotary_pos_emb.shape[0] > 2:
                        position_embeddings = rotary_pos_emb[:2]
                    else:
                        position_embeddings = rotary_pos_emb
                else:
                    position_embeddings = rotary_pos_emb

            cos = None
            sin = None
            is_tuple_or_list = False
            if position_embeddings is not None:
                if isinstance(position_embeddings, (list, tuple)) and len(position_embeddings) == 2:
                    cos, sin = position_embeddings
                    is_tuple_or_list = True
                elif hasattr(position_embeddings, "shape") and position_embeddings.shape[0] == 2:
                    cos = position_embeddings[0]
                    sin = position_embeddings[1]

            if cos is not None and sin is not None:
                if hasattr(self, "attn") and hasattr(cos, "shape"):
                    head_dim = None
                    if hasattr(self.attn, "head_dim"):
                        head_dim = self.attn.head_dim
                    elif hasattr(self.attn, "embed_dim") and hasattr(self.attn, "num_heads"):
                        head_dim = self.attn.embed_dim // self.attn.num_heads
                    if head_dim is not None and cos.shape[-1] != head_dim:
                        if head_dim == cos.shape[-1] * 2:
                            cos = torch.cat([cos, cos], dim=-1)
                            sin = torch.cat([sin, sin], dim=-1)
                            if is_tuple_or_list:
                                position_embeddings = (cos, sin)
                            else:
                                position_embeddings = torch.stack([cos, sin], dim=0)

            return original_forward(self, hidden_states, position_embeddings=position_embeddings, rotary_pos_emb=rotary_pos_emb, **kwargs)

        Qwen2VLVisionBlock.forward = patched_forward
        print("Patched Qwen2VLVisionBlock.forward to accept rotary_pos_emb")

    patch_qwen2_vl_vision_block()
except ImportError:
    print("Could not import Qwen2VLVisionBlock for patching. If you are using Qwen2-VL based models, this might cause issues.")
except Exception as e:
    print(f"Error patching Qwen2VLVisionBlock: {e}")


def load_baichuan_omni_model(model_path: str, device: str):
    cache_dir = os.path.join(os.path.dirname(model_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.training = False
    model.bind_processor(tokenizer, training=False, relative_path=cache_dir)
    return model, tokenizer


def get_special_tokens(model, tokenizer):
    tokens = {}
    if hasattr(model.config, "video_config"):
        tokens["video_start"] = tokenizer.convert_ids_to_tokens(model.config.video_config.video_start_token_id)
        tokens["video_end"] = tokenizer.convert_ids_to_tokens(model.config.video_config.video_end_token_id)
        tokens["image_start"] = tokenizer.convert_ids_to_tokens(model.config.video_config.image_start_token_id)
        tokens["image_end"] = tokenizer.convert_ids_to_tokens(model.config.video_config.image_end_token_id)
    if hasattr(model.config, "audio_config"):
        tokens["audio_start"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_start_token_id)
        tokens["audio_end"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audio_end_token_id)
        tokens["audiogen_start"] = tokenizer.convert_ids_to_tokens(model.config.audio_config.audiogen_start_token_id)
    tokens.setdefault("video_start", "<video_start_baichuan>")
    tokens.setdefault("video_end", "<video_end_baichuan>")
    tokens.setdefault("image_start", "<image_start_baichuan>")
    tokens.setdefault("image_end", "<image_end_baichuan>")
    tokens.setdefault("audio_start", "<audio_start_baichuan>")
    tokens.setdefault("audio_end", "<audio_end_baichuan>")
    return tokens


def _resolve_path(p: str | None, base_dir: str | None):
    if p is None:
        return None
    if base_dir and not os.path.isabs(p):
        return os.path.abspath(os.path.join(base_dir, p))
    return os.path.abspath(p)


def construct_baichuan_conversation(item: Dict[str, Any], special_tokens: Dict[str, str], base_dir: str | None, system_prompt: str):
    modality = item.get("modality", "")
    content = ""

    img_path = _resolve_path(item.get("image") or item.get("image_path"), base_dir)
    if img_path and "image" in modality:
        content += f"{special_tokens['image_start']}" + json.dumps({"local": img_path}, ensure_ascii=False) + f"{special_tokens['image_end']}"

    vid_path = _resolve_path(item.get("video") or item.get("video_path"), base_dir)
    if vid_path and "video" in modality:
        content += f"{special_tokens['video_start']}" + json.dumps({"local": vid_path}, ensure_ascii=False) + f"{special_tokens['video_end']}"

    aud_path = _resolve_path(item.get("audio") or item.get("audio_path"), base_dir)
    if aud_path and "audio" in modality:
        content += f"{special_tokens['audio_start']}" + json.dumps({"path": aud_path}, ensure_ascii=False) + f"{special_tokens['audio_end']}"

    text_field = item.get("text_instruction") or item.get("text", "")
    content += text_field

    return f"<B_SYS>{system_prompt}<C_Q>{content}<C_A>"


def prepare_model_inputs(model, processor_outputs):
    inputs = {"input_ids": processor_outputs.input_ids.to(model.device)}
    if processor_outputs.attention_mask is not None:
        inputs["attention_mask"] = processor_outputs.attention_mask.to(model.device)
    if getattr(processor_outputs, "images", None) is not None:
        inputs["images"] = [torch.tensor(img, dtype=torch.float32).to(model.device) for img in processor_outputs.images]
    if getattr(processor_outputs, "videos", None) is not None:
        inputs["videos"] = [torch.tensor(v, dtype=torch.float32).to(model.device) for v in processor_outputs.videos]
    if getattr(processor_outputs, "audios", None) is not None:
        inputs["audios"] = processor_outputs.audios.to(model.device)
    for key in ["patch_nums", "images_grid", "videos_patch_nums", "videos_grid", "encoder_length", "bridge_length"]:
        val = getattr(processor_outputs, key, None)
        if val is not None:
            inputs[key] = val.to(model.device) if hasattr(val, "to") else val
    return inputs


def select_layer_path(model):
    candidates = ["model.model.layers", "model.layers"]
    modules = dict(model.named_modules())
    for path in candidates:
        if f"{path}.0" in modules:
            return path
    raise ValueError("Could not locate transformer layers for hook registration.")


# -----------------------------------------------------------------------------
# Safety Adapter Definition
# -----------------------------------------------------------------------------
class SafetyAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 1)  # scalar coefficient k
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h)


# -----------------------------------------------------------------------------
# Dataset Definition
# -----------------------------------------------------------------------------
class SafetyDataset(Dataset):
    def __init__(self, data_file, data_root: str | None = None):
        self.data = []
        raw_items: List[Dict[str, Any]] = []
        with open(data_file, 'r') as f:
            try:
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    raw_items = raw_data
                elif isinstance(raw_data, dict):
                    for key in ["text_train", "cross_modal_test", "train", "test"]:
                        if key in raw_data and isinstance(raw_data[key], list):
                            raw_items.extend(raw_data[key])
                    if not raw_items:
                        for v in raw_data.values():
                            if isinstance(v, list):
                                raw_items.extend(v)
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    if line.strip():
                        raw_items.append(json.loads(line))

        harmful_items = [item for item in raw_items if item.get("type", "harmful") == "harmful"]
        safe_items = [item for item in raw_items if item.get("type", "harmful") == "safe"]
        print(f"Loaded {len(harmful_items)} harmful and {len(safe_items)} safe items.")

        if len(safe_items) > 0 and len(harmful_items) > 0:
            multiplier = len(harmful_items) // len(safe_items)
            remainder = len(harmful_items) % len(safe_items)
            balanced_safe = safe_items * multiplier + safe_items[:remainder]
            self.data = harmful_items + balanced_safe
        else:
            self.data = raw_items

        np.random.shuffle(self.data)
        print(f"Final balanced dataset size: {len(self.data)}")
        self.data_root = data_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


# -----------------------------------------------------------------------------
# Refusal Vector Calculation
# -----------------------------------------------------------------------------

def compute_refusal_vectors(model, tokenizer, special_tokens, data_file, target_layers, device, output_dir, data_root: str | None):
    print(f"Computing refusal vectors for layers {target_layers}...")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_train = data.get("text_train", []) if isinstance(data, dict) else []
    cross_modal_test = data.get("cross_modal_test", []) if isinstance(data, dict) else []

    hook_manager = HookManager(model)
    layer_path = select_layer_path(model)
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    vectors = {layer: {} for layer in target_layers}

    print("Processing Text Data for vectors...")
    text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
    for item in tqdm(text_train[:100], desc="Text"):
        convo_item = {"text": item["text"], "modality": "text"}
        full_input = construct_baichuan_conversation(
            convo_item,
            special_tokens,
            base_dir=data_root,
            system_prompt="You are a helpful assistant."
        )
        processed = model.processor([full_input])
        model_inputs = prepare_model_inputs(model, processed)
        with torch.no_grad():
            model(**model_inputs)
        for layer in target_layers:
            if layer in hook_manager.activations:
                act = hook_manager.activations[layer][:, -1, :].cpu()
                text_acts[layer][item.get("type", "harmful")].append(act)
        hook_manager.activations = {}

    for layer in target_layers:
        if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
            vectors[layer]["text"] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    print("Processing Multi-Modal Data for vectors...")
    mm_acts = {m: {layer: {"safe": [], "harmful": []} for layer in target_layers} for m in ["image", "audio", "video"]}
    counts = {m: {"safe": 0, "harmful": 0} for m in ["image", "audio", "video"]}

    for item in tqdm(cross_modal_test, desc="MM"):
        modality = item.get("modality", "")
        label = item.get("label", "harmful")
        if modality not in mm_acts or counts[modality][label] >= 50:
            continue
        convo_item = {
            "modality": modality,
            "text": item.get("text_instruction", ""),
            "text_instruction": item.get("text_instruction", ""),
            "image": item.get("image_path") or item.get("image"),
            "audio": item.get("audio_path") or item.get("audio"),
            "video": item.get("video_path") or item.get("video"),
        }
        full_input = construct_baichuan_conversation(
            convo_item,
            special_tokens,
            base_dir=data_root,
            system_prompt="You are a helpful assistant."
        )
        processed = model.processor([full_input])
        model_inputs = prepare_model_inputs(model, processed)
        with torch.no_grad():
            model(**model_inputs)
        for layer in target_layers:
            if layer in hook_manager.activations:
                act = hook_manager.activations[layer][:, -1, :].cpu()
                mm_acts[modality][layer][label].append(act)
        hook_manager.activations = {}
        counts[modality][label] += 1

    for m in mm_acts:
        for layer in target_layers:
            if mm_acts[m][layer]["safe"] and mm_acts[m][layer]["harmful"]:
                vectors[layer][m] = calculate_refusal_vector(mm_acts[m][layer]["safe"], mm_acts[m][layer]["harmful"]).cpu()

    results = {}
    os.makedirs(output_dir, exist_ok=True)
    for layer in target_layers:
        layer_vecs = vectors[layer]
        if len(layer_vecs) < 1:
            print(f"Not enough vectors for layer {layer}")
            continue
        X = torch.stack([v for v in layer_vecs.values()]).float().numpy()
        if len(X) >= 1:
            pca = PCA(n_components=1)
            pca.fit(X)
            pc1 = torch.tensor(pca.components_[0], dtype=torch.float32)
            save_path = os.path.join(output_dir, f"canonical_refusal_vector_layer_{layer}.pt")
            torch.save(pc1, save_path)
            results[layer] = pc1
            print(f"Computed and saved PC1 for Layer {layer}")
    return results


# -----------------------------------------------------------------------------
# Training Logic
# -----------------------------------------------------------------------------

def train_adapter(args):
    print(f"Loading Baichuan-Omni model from {args.model_path}...")
    model, tokenizer = load_baichuan_omni_model(args.model_path, args.device)
    special_tokens = get_special_tokens(model, tokenizer)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    target_layers = [int(x) for x in args.target_layers.split(',')]
    unified_refusal_vectors = {}

    missing_layers = []
    for layer in target_layers:
        path = os.path.join(args.refusal_vector_dir, f"canonical_refusal_vector_layer_{layer}.pt")
        if os.path.exists(path):
            print(f"Loading vector for layer {layer} from {path}")
            unified_refusal_vectors[layer] = torch.load(path).to(args.device)
        else:
            print(f"Vector for layer {layer} not found.")
            missing_layers.append(layer)

    if missing_layers:
        print(f"Calculating missing vectors for layers: {missing_layers}")
        computed = compute_refusal_vectors(
            model,
            tokenizer,
            special_tokens,
            args.data_file,
            missing_layers,
            args.device,
            args.refusal_vector_dir,
            args.data_root,
        )
        for layer, vec in computed.items():
            unified_refusal_vectors[layer] = vec.to(args.device)

    for layer in unified_refusal_vectors:
        unified_refusal_vectors[layer] = unified_refusal_vectors[layer].to(args.device).float()
        # unified_refusal_vectors[layer] = unified_refusal_vectors[layer] / (torch.norm(unified_refusal_vectors[layer]) + 1e-6)

    hook_manager = HookManager(model)
    layer_path = select_layer_path(model)
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
    adapters = nn.ModuleDict({
        str(layer): SafetyAdapter(hidden_dim, bottleneck_dim=args.bottleneck_dim)
        for layer in target_layers
    }).to(args.device)
    optimizer = optim.AdamW(adapters.parameters(), lr=args.lr)

    dataset = SafetyDataset(args.data_file, data_root=args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Loss tracking
    step_history: List[int] = []
    total_history: List[float] = []
    harmful_history: List[float] = []
    safe_history: List[float] = []

    def save_loss_plot(step: int, suffix: str):
        if not step_history:
            return
        plt.figure(figsize=(8, 5))
        plt.plot(step_history, total_history, label="total", color="blue")
        plt.plot(step_history, harmful_history, label="harmful", color="red")
        plt.plot(step_history, safe_history, label="safe", color="green")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"Training Loss ({suffix})")
        plt.legend()
        plt.tight_layout()
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"loss_{suffix}_step_{step}.png")
        plt.savefig(out_path)
        plt.close()

    global_step = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        steps = 0
        batch_counter = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            harmful_losses = []
            safe_losses = []

            for item in batch:
                modality = item.get("modality", "text")
                convo_item = {
                    "modality": modality,
                    "text": item.get("text", ""),
                    "text_instruction": item.get("text", ""),
                    "image": item.get("image") or item.get("image_path"),
                    "audio": item.get("audio") or item.get("audio_path"),
                    "video": item.get("video") or item.get("video_path"),
                }
                full_input = construct_baichuan_conversation(
                    convo_item,
                    special_tokens,
                    base_dir=args.data_root,
                    system_prompt="You are a helpful assistant.",
                )
                try:
                    processed = model.processor([full_input])
                    model_inputs = prepare_model_inputs(model, processed)
                except Exception:
                    continue

                with torch.no_grad():
                    model(**model_inputs)

                label = item.get("type", "safe")
                item_loss = 0
                for layer in target_layers:
                    if layer not in unified_refusal_vectors or layer not in hook_manager.activations:
                        continue
                    current_h = hook_manager.activations[layer][0, -1, :].detach().to(args.device).float()
                    adapter = adapters[str(layer)]
                    k = adapter(current_h).squeeze()
                    pc1 = unified_refusal_vectors[layer]
                    refusal_norm = torch.norm(pc1) + 1e-6
                    pc1_dir = pc1 / refusal_norm
                    delta_h = k * pc1_dir
                    final_h = current_h + delta_h
                    proj_val = torch.dot(final_h.to(pc1.dtype), pc1_dir)
                    print(f"Layer {layer} | label: {label} | k: {k.item():.4f} | proj: {proj_val.item():.4f} | norm_raw: {refusal_norm.item():.4f}")

                    if label == "harmful":
                        target_threshold = 0.8 * refusal_norm
                        loss_proj = torch.relu(target_threshold - proj_val) / refusal_norm
                        loss_reg = 0.01 * torch.abs(k)
                        loss = loss_proj + loss_reg
                    else:
                        safe_pull = 0.4 * refusal_norm
                        loss_safe = torch.relu(proj_val + safe_pull) / refusal_norm
                        loss_reg = 0.02 * torch.abs(k)
                        loss = loss_safe + loss_reg
                    item_loss += loss

                hook_manager.activations = {}
                if label == "harmful" and item_loss != 0:
                    harmful_losses.append(item_loss)
                elif label != "harmful" and item_loss != 0:
                    safe_losses.append(item_loss)

            if harmful_losses or safe_losses:
                batch_loss = 0
                if harmful_losses:
                    batch_loss += torch.stack(harmful_losses).mean()
                if safe_losses:
                    batch_loss += torch.stack(safe_losses).mean()
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                steps += 1
                batch_counter += 1

                cur_h = torch.stack(harmful_losses).mean().item() if harmful_losses else float('nan')
                cur_s = torch.stack(safe_losses).mean().item() if safe_losses else float('nan')
                cur_t = batch_loss.item()
                global_step += 1
                step_history.append(global_step)
                total_history.append(cur_t)
                harmful_history.append(cur_h)
                safe_history.append(cur_s)

                if global_step % args.plot_interval == 0:
                    save_loss_plot(global_step, f"epoch{epoch+1}")

                # Save checkpoint every 300 batches
                if batch_counter % 300 == 0:
                    os.makedirs(args.output_dir, exist_ok=True)
                    ckpt_path = os.path.join(args.output_dir, f"safety_adapter_step_{global_step}.pt")
                    torch.save(adapters.state_dict(), ckpt_path)
                    print(f"Saved interim checkpoint to {ckpt_path}")
            progress_bar.set_postfix({"loss": total_loss / steps if steps > 0 else 0})

        save_path = os.path.join(args.output_dir, f"safety_adapter_epoch_{epoch}.pt")
        torch.save(adapters.state_dict(), save_path)
        print(f"Saved adapters to {save_path}")

        save_loss_plot(global_step, f"epoch{epoch+1}_end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/adapter_baichuan")
    parser.add_argument("--refusal_vector_dir", type=str, default="output/vector_analysis_baichuan", help="Directory containing pre-calculated PC1 vectors")
    parser.add_argument("--target_layers", type=str, default="19", help="Comma separated list of layers, e.g. '18,19,20'")
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=None, help="Optional base dir to resolve relative media paths")
    parser.add_argument("--plot_interval", type=int, default=200, help="Save loss curves every N training steps")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_adapter(args)
