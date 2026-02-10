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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import HookManager, calculate_refusal_vector
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# MiniCPM-o specific imports and patches (adapted from safe_eval/unified_eval_minicpm_o.py)
# -----------------------------------------------------------------------------
import transformers.models.whisper.modeling_whisper as whisper_mod
from transformers.cache_utils import DynamicCache
from transformers import AutoModel, AutoTokenizer

# Disable tokenizers parallelism to avoid deadlocks in multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optional multimedia dependencies
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import librosa
except ImportError:
    librosa = None
try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:
    VideoReader = None
    decord_cpu = None

# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Patch DynamicCache.seen_tokens if missing (for newer transformers versions)
if not hasattr(DynamicCache, "seen_tokens"):
    try:
        @property
        def seen_tokens(self):
            return self.get_seq_length()
        DynamicCache.seen_tokens = seen_tokens
    except Exception:
        pass

# Patch WHISPER_ATTENTION_CLASSES if missing (for newer transformers versions)
if not hasattr(whisper_mod, "WHISPER_ATTENTION_CLASSES"):
    try:
        attention_classes = {"eager": whisper_mod.WhisperAttention}
        if hasattr(whisper_mod, "WhisperSdpaAttention"):
            attention_classes["sdpa"] = whisper_mod.WhisperSdpaAttention
        else:
            attention_classes["sdpa"] = whisper_mod.WhisperAttention
        if hasattr(whisper_mod, "WhisperFlashAttention2"):
            attention_classes["flash_attention_2"] = whisper_mod.WhisperFlashAttention2
        else:
            attention_classes["flash_attention_2"] = whisper_mod.WhisperAttention
        whisper_mod.WHISPER_ATTENTION_CLASSES = attention_classes
    except Exception:
        pass

# Patch WhisperAttention.forward to ensure it returns 3 values
if hasattr(whisper_mod, "WhisperAttention"):
    original_forward = whisper_mod.WhisperAttention.forward
    def patched_forward(self, *args, **kwargs):
        outputs = original_forward(self, *args, **kwargs)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            return outputs + (None,)
        return outputs
    whisper_mod.WhisperAttention.forward = patched_forward


# -----------------------------------------------------------------------------
# Model helpers
# -----------------------------------------------------------------------------

def load_minicpm_o_model(model_path: str, device: str):
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def encode_video(video_path: str):
    if VideoReader is None:
        raise ImportError("decord is required for video processing. Please install it.")
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    max_frames = 64
    sample_fps = round(vr.get_avg_fps() / 1)
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames:
        gap = len(frame_idx) / max_frames
        frame_idx = [frame_idx[int(i * gap + gap / 2)] for i in range(max_frames)]
    frames = vr.get_batch(frame_idx).asnumpy()
    if Image is None:
        raise ImportError("Pillow is required for image processing. Please install it.")
    return [Image.fromarray(v.astype("uint8")) for v in frames]


def construct_conversation(item: Dict[str, Any], base_dir: str | None = None, use_self_reminder: bool = False):
    content = []
    modality = item.get("modality", "")

    # Image
    if "image" in modality and item.get("image"):
        if Image is None:
            raise ImportError("Pillow is required for image processing. Please install it.")
        img_path = item.get("image")
        if base_dir and not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        content.append(image)

    # Video
    if "video" in modality and item.get("video"):
        vid_path = item.get("video")
        if base_dir and not os.path.isabs(vid_path):
            vid_path = os.path.join(base_dir, vid_path)
        frames = encode_video(vid_path)
        content.extend(frames)

    # Audio
    if "audio" in modality and item.get("audio"):
        if librosa is None:
            raise ImportError("librosa is required for audio processing. Please install it.")
        aud_path = item.get("audio")
        if base_dir and not os.path.isabs(aud_path):
            aud_path = os.path.join(base_dir, aud_path)
        audio, _ = librosa.load(aud_path, sr=16000, mono=True)
        content.append(audio)

    # Text
    if "text" in modality and (item.get("text") or item.get("text_instruction")):
        text = item.get("text_instruction") or item.get("text", "")
        if use_self_reminder:
            text = (
                "You should be a responsible assistant and should not generate harmful or misleading content! "
                "Please answer the following user query in a responsible way.\n"
                f"{text}\n Remember, you should be a responsible assistant and should not generate harmful or misleading content!"
            )
        content.append(text)

    msgs = [{"role": "user", "content": content}]
    return msgs


def select_layer_path(model):
    candidates = [
        "model.model.layers",  # common
        "model.layers",         # alternative
        "llm.model.layers",     # some MiniCPM variants
    ]
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
            nn.Linear(bottleneck_dim, 1)  # output a scalar coefficient k
        )
        # start from no intervention (k=0)
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

def compute_refusal_vectors(model, tokenizer, data_file, target_layers, device, output_dir, data_root: str | None):
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
        convo_item = {"text": item.get("text", ""), "modality": "text"}
        msgs = construct_conversation(convo_item, base_dir=data_root, use_self_reminder=False)
        with torch.no_grad():
            model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                max_new_tokens=1,
                generate_audio=False,
            )
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
        msgs = construct_conversation(convo_item, base_dir=data_root, use_self_reminder=False)
        with torch.no_grad():
            model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                max_new_tokens=1,
                generate_audio=False,
            )
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
    print(f"Loading MiniCPM-o model from {args.model_path}...")
    model, tokenizer = load_minicpm_o_model(args.model_path, args.device)

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
        # Keep vectors normalized so k exclusively controls magnitude
        unified_refusal_vectors[layer] = unified_refusal_vectors[layer] / (torch.norm(unified_refusal_vectors[layer]) + 1e-6)

    hook_manager = HookManager(model)
    layer_path = select_layer_path(model)
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    hidden_dim = model.config.hidden_size if hasattr(model.config, "hidden_size") else getattr(model.config, "n_embd", None)
    if hidden_dim is None:
        raise ValueError("Could not determine hidden dimension from model config.")

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
                msgs = construct_conversation(convo_item, base_dir=args.data_root, use_self_reminder=False)

                try:
                    with torch.no_grad():
                        model.chat(
                            msgs=msgs,
                            tokenizer=tokenizer,
                            sampling=False,
                            max_new_tokens=1,
                            generate_audio=False,
                        )
                except Exception:
                    continue

                label = item.get("type", "safe")
                item_loss = 0
                for layer in target_layers:
                    if layer not in unified_refusal_vectors or layer not in hook_manager.activations:
                        continue
                    current_h = hook_manager.activations[layer][0, -1, :].detach().to(args.device).float()
                    adapter = adapters[str(layer)]
                    k = adapter(current_h).squeeze()  # scalar coefficient
                    pc1 = unified_refusal_vectors[layer]
                    refusal_norm = torch.norm(pc1) + 1e-6
                    pc1_dir = pc1 / refusal_norm
                    delta_h = k * pc1_dir  # steer only along refusal direction
                    final_h = current_h + delta_h
                    proj_val = torch.dot(final_h.to(pc1.dtype), pc1_dir)
                    print(f"Layer {layer} | label: {label} | k: {k.item():.4f} | proj: {proj_val.item():.4f} | norm_raw: {refusal_norm.item():.4f}")

                    if label == "harmful":
                        target_threshold = 10.0 * refusal_norm
                        loss_proj = torch.relu(target_threshold - proj_val) / refusal_norm
                        loss_reg = 0.005 * torch.abs(k)
                        loss = loss_proj + loss_reg
                    else:
                        safe_pull = 3.0 * refusal_norm
                        loss_safe = torch.relu(proj_val + safe_pull) / refusal_norm
                        loss_reg = 0.01 * torch.abs(k)
                        loss = loss_safe + loss_reg
                    item_loss += loss

                hook_manager.activations = {}
                # Always append to maintain balanced batches; ensure tensor dtype
                if item_loss == 0:
                    item_loss = torch.tensor(0.0, device=args.device)
                if label == "harmful":
                    harmful_losses.append(item_loss)
                else:
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
                # Record losses
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

        # Save per-epoch loss plot snapshot
        save_loss_plot(global_step, f"epoch{epoch+1}_end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/adapter_minicpm_o")
    parser.add_argument("--refusal_vector_dir", type=str, default="output/vector_analysis_minicpm_o", help="Directory containing pre-calculated PC1 vectors")
    parser.add_argument("--target_layers", type=str, default="19", help="Comma separated list of layers, e.g. '18,19,20'")
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=None, help="Optional base dir to resolve relative media paths")
    parser.add_argument("--plot_interval", type=int, default=200, help="Save loss curves every N training steps")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_adapter(args)
