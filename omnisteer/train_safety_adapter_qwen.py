import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from typing import List
from utils import load_model, HookManager, calculate_refusal_vector
from sklearn.decomposition import PCA
from qwen_omni_utils import process_mm_info

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Safety Adapter Definition
# -----------------------------------------------------------------------------
class SafetyAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 1)  # scalar coefficient k
        )
        # start from no intervention
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return self.net(h)

# -----------------------------------------------------------------------------
# 2. Dataset Definition
# -----------------------------------------------------------------------------
class SafetyDataset(Dataset):
    def __init__(self, data_file, processor, model_device):
        self.data = []
        raw_items = []
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
        
        # --- Balanced Sampling Implementation ---
        harmful_items = [item for item in raw_items if item.get("type", "harmful") == "harmful"]
        safe_items = [item for item in raw_items if item.get("type", "harmful") == "safe"]
        
        print(f"Loaded {len(harmful_items)} harmful and {len(safe_items)} safe items.")
        
        # Oversample safe items to match harmful items count for 1:1 balance
        if len(safe_items) > 0 and len(harmful_items) > 0:
            multiplier = len(harmful_items) // len(safe_items)
            remainder = len(harmful_items) % len(safe_items)
            balanced_safe = safe_items * multiplier + safe_items[:remainder]
            self.data = harmful_items + balanced_safe
        else:
            self.data = raw_items
            
        # Shuffle the combined data thoroughly
        np.random.shuffle(self.data)
        print(f"Final balanced dataset size: {len(self.data)}")
        
        self.processor = processor
        self.device = model_device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def collate_fn(self, batch):
        return batch

# -----------------------------------------------------------------------------
# 3. Vector Calculation Logic (Fallback)
# -----------------------------------------------------------------------------
def compute_refusal_vectors(model, processor, data_file, target_layers, device, output_dir):
    print(f"Computing refusal vectors for layers {target_layers}...")
    
    # Load data specifically for vector calculation (supports .json and .jsonl)
    data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        try:
            # Try to load as a single JSON document first
            f.seek(0)
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback: treat file as JSONL (one JSON object per line)
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

    # Ensure we have the right structure or fallback
    if isinstance(data, list):
        # Assume list contains mixed data, separate if possible or use as is
        # For refusal vector, we need safe/harmful pairs.
        # This might be tricky if the list is just training samples.
        # We'll assume the user provides the 'experiment_vector_analysis' compatible file
        # if they expect calculation to happen.
        print("Warning: Data file is a list. Assuming it contains 'text_train' and 'cross_modal_test' compatible items.")
        # This part is fragile without strict schema, but let's try to proceed
        text_train = [x for x in data if x.get("modality", "text") == "text"]
        cross_modal_test = [x for x in data if x.get("modality", "text") != "text"]
    else:
        text_train = data.get("text_train", [])
        cross_modal_test = data.get("cross_modal_test", [])

    hook_manager = HookManager(model)
    layer_path = "thinker.model.layers"
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    vectors = {layer: {} for layer in target_layers}
    
    # --- Text ---
    print("Processing Text Data...")
    text_acts = {layer: {"safe": [], "harmful": []} for layer in target_layers}
    for item in tqdm(text_train[:100], desc="Text"): # Limit for speed
        content = [{"type": "text", "text": item["text"]}]
        conversation = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}, {"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
        
        for layer in target_layers:
            if layer in hook_manager.activations:
                act = hook_manager.activations[layer][:, -1, :].cpu()
                text_acts[layer][item["type"]].append(act)
        hook_manager.activations = {}

    for layer in target_layers:
        if text_acts[layer]["safe"] and text_acts[layer]["harmful"]:
            vectors[layer]["text"] = calculate_refusal_vector(text_acts[layer]["safe"], text_acts[layer]["harmful"]).cpu()

    # --- Multi-Modal ---
    print("Processing Multi-Modal Data...")
    mm_acts = {m: {layer: {"safe": [], "harmful": []} for layer in target_layers} for m in ["image", "audio", "video"]}
    counts = {m: {"safe": 0, "harmful": 0} for m in ["image", "audio", "video"]}
    
    for item in tqdm(cross_modal_test, desc="MM"):
        modality = item.get("modality", "")
        label = item.get("label", "harmful")
        if modality not in mm_acts or counts[modality][label] >= 50: continue # Limit
        
        content = []
        if modality == "image": content.append({"type": "image", "image": item["image"]})
        elif modality == "audio": content.append({"type": "audio", "audio": item["audio"]})
        elif modality == "video": content.append({"type": "video", "video": item["video"]})
        
        conversation = [{"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]}, {"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, False)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
            
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

    # --- PCA & Save ---
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for layer in target_layers:
        layer_vecs = vectors[layer]
        if len(layer_vecs) < 1: 
            print(f"Not enough vectors for layer {layer}")
            continue
            
        # Use whatever modalities we have
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
# 4. Training Logic
# -----------------------------------------------------------------------------
def train_adapter(args):
    # A. Load Model & Processor
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)
    
    # Freeze the main model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # B. Prepare PC1 Vectors (Load or Compute)
    target_layers = [int(x) for x in args.target_layers.split(',')]
    unified_refusal_vectors = {}
    
    # Check which are missing
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
        computed = compute_refusal_vectors(model, processor, args.data_file, missing_layers, args.device, args.refusal_vector_dir)
        for layer, vec in computed.items():
            unified_refusal_vectors[layer] = vec.to(args.device)
            
    # Normalize vectors and ensure they are in float32 for loss calculation
    for layer in unified_refusal_vectors:
        unified_refusal_vectors[layer] = unified_refusal_vectors[layer].to(args.device).float()

    # C. Initialize Adapters (One per layer)
    hook_manager = HookManager(model)
    layer_path = "thinker.model.layers"
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)

    hidden_dim = model.config.hidden_size if hasattr(model.config, 'hidden_size') else 3584
    
    # Use ModuleDict to manage multiple adapters
    # Keep adapters in float32 for training stability
    adapters = nn.ModuleDict({
        str(layer): SafetyAdapter(hidden_dim, bottleneck_dim=args.bottleneck_dim) 
        for layer in target_layers
    }).to(args.device)
    
    optimizer = optim.AdamW(adapters.parameters(), lr=args.lr)

    # D. Load Data
    dataset = SafetyDataset(args.data_file, processor, args.device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # E. Training Loop
    print("Starting training...")

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
                # 1. Prepare Input
                content = []
                modality = item.get("modality", "text")
                
                if "image" in modality:
                    content.append({"type": "image", "image": item["image"]})
                if "video" in modality:
                    content.append({"type": "video", "video": item["video"]})
                if "audio" in modality:
                    content.append({"type": "audio", "audio": item["audio"]})
                
                if "text" in modality:
                    content.append({"type": "text", "text": item["text"]})
                
                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                    {"role": "user", "content": content}
                ]
                
                try:
                    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                    audios, images, videos = process_mm_info(conversation, True) 
                    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                except Exception as e:
                    # print(f"Error processing item {item.get('id')}: {e}")
                    continue

                # 2. Forward Pass
                with torch.no_grad():
                    model.thinker(**inputs)
                
                # 3. Calculate Loss for each layer
                item_loss = 0
                label = item.get("type", "safe")
                
                for layer in target_layers:
                    if layer not in unified_refusal_vectors: continue
                    
                    # Get activation and move to the correct device
                    current_h = hook_manager.activations[layer][0, -1, :].detach().to(args.device).float()
                    
                    # Adapter forward
                    adapter = adapters[str(layer)]
                    k = adapter(current_h).squeeze()  # scalar
                    
                    # Get the target refusal vector (SVD vector)
                    pc1 = unified_refusal_vectors[layer]
                    refusal_norm = torch.norm(pc1) + 1e-6  # keep raw norm for scale targets
                    pc1_dir = pc1 / refusal_norm
                    
                    delta_h = k * pc1_dir  # steer only along refusal direction
                    final_h = current_h + delta_h
                    
                    # Calculate projection of final state onto the refusal direction
                    proj_val = torch.dot(final_h.to(pc1.dtype), pc1_dir)
                    print(f"Layer {layer} | label: {label} | k: {k.item():.4f} | proj: {proj_val.item():.4f} | norm_raw: {refusal_norm.item():.4f}")
                    
                    if label == "harmful":
                        target_threshold = 0.3 * refusal_norm
                        loss_proj = torch.relu(target_threshold - proj_val) / refusal_norm
                        loss_reg = 0.01 * torch.abs(k)
                        loss = loss_proj + loss_reg
                    else:
                        safe_pull = 0.5 * refusal_norm
                        loss_safe = torch.relu(proj_val + safe_pull) / refusal_norm
                        loss_reg = 0.05 * torch.abs(k)
                        loss = loss_safe + loss_reg
                    
                    item_loss += loss
                
                hook_manager.activations = {} # Clear buffer
                
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
        
        # Save checkpoint
        save_path = os.path.join(args.output_dir, f"safety_adapter_epoch_{epoch}.pt")
        torch.save(adapters.state_dict(), save_path)
        print(f"Saved adapters to {save_path}")

        save_loss_plot(global_step, f"epoch{epoch+1}_end")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/adapter")
    parser.add_argument("--refusal_vector_dir", type=str, default="output/vector_analysis", help="Directory containing the pre-calculated PC1 vector .pt files")
    parser.add_argument("--target_layers", type=str, default="19", help="Comma separated list of layers, e.g. '18,19,20'")
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--plot_interval", type=int, default=200, help="Save loss curves every N training steps")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_adapter(args)
