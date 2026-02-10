import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from utils import HookManager, calculate_refusal_vector

# MiniCPM-o specific imports and patches (adapted from safe_eval/unified_eval_minicpm_o.py)
import logging
import warnings
import sys
from transformers import AutoModel, AutoTokenizer
import transformers.models.whisper.modeling_whisper as whisper_mod
from transformers.cache_utils import DynamicCache

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


def load_minicpm_o_model(model_path: str, device: str):
    """
    Load MiniCPM-o 2.6 model with vision/audio initialized.
    """
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


def construct_conversation(item: dict, base_dir: str | None = None, use_self_reminder: bool = False):
    """
    Build the conversation messages list expected by MiniCPM-o's model.chat.
    Content is a list of [Image/VideoFrames/Audio, Text].
    """
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


def run_vector_analysis(args):
    print(f"Loading MiniCPM-o model from {args.model_path}...")
    model, tokenizer = load_minicpm_o_model(args.model_path, args.device)

    hook_manager = HookManager(model)
    target_layers = list(range(10, 21))
    layer_path = select_layer_path(model)
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    with open(args.data_file, "r") as f:
        data = json.load(f)

    vectors = {layer: {} for layer in target_layers}

    # ----------------------
    # Text refusal vectors
    # ----------------------
    print("Calculating Text Refusal Vectors...")
    text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
    train_subset = data["text_train"]

    for item in tqdm(train_subset, desc="Text Data"):
        convo_item = {
            "text": item["text"],
            "modality": "text",
        }
        msgs = construct_conversation(convo_item, base_dir=args.data_root, use_self_reminder=False)
        with torch.no_grad():
            # model.chat handles preprocessing; hook captures hidden states
            model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                max_new_tokens=1,
                generate_audio=False,
            )
        for layer in target_layers:
            act = hook_manager.activations[layer][:, -1, :].cpu()
            text_acts[layer][item["type"]].append(act)
        hook_manager.activations = {}

    for layer in target_layers:
        if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
            vectors[layer]["text"] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    # ----------------------
    # Multi-modal refusal vectors
    # ----------------------
    target_modalities = ["image", "audio", "video", "text+image", "text+audio", "text+video"]
    mm_acts = {m: {layer: {"safe": [], "harmful": []} for layer in target_layers} for m in target_modalities}
    max_samples = 200
    counts = {m: {"safe": 0, "harmful": 0} for m in target_modalities}

    print("Calculating Multi-Modal Refusal Vectors...")
    for item in tqdm(data["cross_modal_test"], desc="MM Data"):
        modality = item.get("modality", "")
        label = item.get("label", "harmful")
        if modality not in target_modalities:
            continue
        if counts[modality][label] >= max_samples:
            continue

        convo_item = {
            "modality": modality,
            "text": item.get("text_instruction", ""),
            "text_instruction": item.get("text_instruction", ""),
            "image": item.get("image_path") or item.get("image"),
            "audio": item.get("audio_path") or item.get("audio"),
            "video": item.get("video_path") or item.get("video"),
        }
        msgs = construct_conversation(convo_item, base_dir=args.data_root, use_self_reminder=False)
        with torch.no_grad():
            model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=False,
                max_new_tokens=1,
                generate_audio=False,
            )
        for layer in target_layers:
            act = hook_manager.activations[layer][:, -1, :].cpu()
            mm_acts[modality][layer][label].append(act)
        hook_manager.activations = {}
        counts[modality][label] += 1

    for m in target_modalities:
        for layer in target_layers:
            s_acts = mm_acts[m][layer]["safe"]
            h_acts = mm_acts[m][layer]["harmful"]
            if s_acts and h_acts:
                vectors[layer][m] = calculate_refusal_vector(s_acts, h_acts).cpu()

    # ----------------------
    # Analysis & visualization (same as original)
    # ----------------------
    print("Analyzing vectors...")
    os.makedirs(args.output_dir, exist_ok=True)
    analysis_results = {}

    for layer in target_layers:
        layer_vecs = vectors[layer]
        if len(layer_vecs) < 1:
            continue

        base_vec = layer_vecs.get("text")
        aligned_vecs = []
        modality_names = list(layer_vecs.keys())

        if base_vec is not None:
            base_norm = torch.norm(base_vec) + 1e-6
            for m in modality_names:
                v = layer_vecs[m]
                _ = torch.dot(v.to(base_vec.dtype), base_vec) / (torch.norm(v) * base_norm)
                aligned_vecs.append(v)
        else:
            aligned_vecs = [layer_vecs[m] for m in modality_names]

        avg_norm = np.mean([torch.norm(v).item() for v in aligned_vecs])
        X = torch.stack(aligned_vecs).float().numpy()

        pca = PCA(n_components=min(len(modality_names), 3))
        pca.fit(X)
        explained_variance = pca.explained_variance_ratio_

        pc1 = pca.components_[0]
        avg_vec = X.mean(axis=0)
        if np.dot(pc1, avg_vec) < 0:
            pc1 = -pc1
        pc1_scaled = pc1 * avg_norm

        pc1_tensor = torch.tensor(pc1_scaled, dtype=torch.float32)
        torch.save(pc1_tensor, os.path.join(args.output_dir, f"pca_refusal_vector_layer_{layer}.pt"))

        pure_keys = ["text", "image", "audio", "video"]
        pure_indices = [i for i, m in enumerate(modality_names) if m in pure_keys]

        if len(pure_indices) > 0:
            pure_aligned_vecs = [aligned_vecs[i] for i in pure_indices]
            mean_vec = torch.stack(pure_aligned_vecs).mean(dim=0)
            torch.save(mean_vec, os.path.join(args.output_dir, f"mean_refusal_vector_layer_{layer}.pt"))

            if "text" in layer_vecs:
                text_vec = layer_vecs["text"]
                sim_mean = torch.dot(mean_vec.to(text_vec.dtype), text_vec) / (torch.norm(mean_vec) * torch.norm(text_vec))
                print(f"Layer {layer}: Pure Mean Vector vs Text CosSim = {sim_mean.item():.4f}")

        if len(pure_indices) > 0:
            X_pure = X[pure_indices]
            _, _, vh = np.linalg.svd(X_pure, full_matrices=False)
            canonical_vec = vh[0]
            if np.dot(canonical_vec, X_pure.mean(axis=0)) < 0:
                canonical_vec = -canonical_vec
            pure_avg_norm = np.mean([torch.norm(layer_vecs[modality_names[i]]).item() for i in pure_indices])
            canonical_vec_scaled = canonical_vec * pure_avg_norm
            canon_tensor = torch.tensor(canonical_vec_scaled, dtype=torch.float32)
            torch.save(canon_tensor, os.path.join(args.output_dir, f"canonical_refusal_vector_layer_{layer}.pt"))

            if "text" in layer_vecs:
                text_vec = layer_vecs["text"].numpy()
                sim = np.dot(canonical_vec, text_vec) / (np.linalg.norm(canonical_vec) * np.linalg.norm(text_vec))
                print(f"Layer {layer}: Canonical (Pure SVD) vs Text CosSim = {sim:.4f}")

        alignment_scores = {}
        for i, m in enumerate(modality_names):
            vec = X[i]
            cos_sim = np.dot(vec, pc1) / (np.linalg.norm(vec) * np.linalg.norm(pc1))
            alignment_scores[m] = float(cos_sim)

        analysis_results[layer] = {
            "explained_variance_ratio": explained_variance.tolist(),
            "pc1_alignment": alignment_scores,
        }

        print(f"Layer {layer}: PC1 Explained Var = {explained_variance[0]:.4f}")

        if pca.n_components_ >= 2:
            X_pca = pca.transform(X)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=modality_names, s=100, style=modality_names)
            for i, txt in enumerate(modality_names):
                plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=12)
                plt.arrow(0, 0, X_pca[i, 0], X_pca[i, 1], color="gray", alpha=0.3, width=0.002)
            plt.title(f"Layer {layer} Refusal Vectors PCA")
            plt.xlabel(f"PC1 ({explained_variance[0]:.2%})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.2%})")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(args.output_dir, f"layer_{layer}_pca.png"))
            plt.close()

    with open(os.path.join(args.output_dir, "vector_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Analysis done. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/vector_analysis_minicpm_o")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default=None, help="Optional base dir for resolving relative media paths")
    args = parser.parse_args()
    run_vector_analysis(args)
