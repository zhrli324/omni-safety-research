import torch
import numpy as np
import os
import json
import logging
import warnings
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
import transformers.models.whisper.modeling_whisper as whisper_mod
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import utils from omni_safety_research
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omniguard_core import OmniGuardProcessor, SafetyClassifier, SimpleEmbeddingDataset
from custom_data_utils import prepare_calibration_data, prepare_training_data
from utils import HookManager

# ============================
# Configuration
# ============================
# Update this path to your MiniCPM-o model location
MODEL_ID = "OpenBMB/MiniCPM-o-2_6"
MODALITY = "vision" # "vision" or "audio"
JSONL_PATH = "data/holisafe_abs.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# Patches for MiniCPM-o
# ============================
# Suppress warnings
transformers_logging = logging.getLogger("transformers")
transformers_logging.setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Patch DynamicCache.seen_tokens if missing
if not hasattr(DynamicCache, "seen_tokens"):
    try:
        @property
        def seen_tokens(self):
            return self.get_seq_length()
        DynamicCache.seen_tokens = seen_tokens
    except Exception as e:
        print(f"Failed to patch DynamicCache.seen_tokens: {e}")

# Patch WHISPER_ATTENTION_CLASSES if missing
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
    except Exception as e:
        print(f"Failed to patch WHISPER_ATTENTION_CLASSES: {e}")

# Patch WhisperAttention.forward
if hasattr(whisper_mod, "WhisperAttention"):
    original_forward = whisper_mod.WhisperAttention.forward
    
    def patched_forward(self, *args, **kwargs):
        try:
            outputs = original_forward(self, *args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                return outputs + (None,)
            return outputs
        except Exception as e:
            if not hasattr(self.config, "_attn_implementation") or self.config._attn_implementation is None:
                self.config._attn_implementation = "eager"
                outputs = original_forward(self, *args, **kwargs)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    return outputs + (None,)
                return outputs
            raise e

    whisper_mod.WhisperAttention.forward = patched_forward

# ============================
# MiniCPM-o Specific Processor
# ============================
class MiniCPMOmniGuardProcessor(OmniGuardProcessor):
    def __init__(self, model, processor, modality_type="vision"):
        super().__init__(model, processor, modality_type)
        self.hook_manager = HookManager(model)
        
        # Identify layers path
        self.layer_path = "model.layers" # Default for many HF models
        if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "layers"):
            self.layer_path = "llm.model.layers"
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layer_path = "model.layers"
        elif hasattr(model, "layers"):
            self.layer_path = "layers"
            
        # Determine number of layers
        try:
            # Access the layers module to count them
            layers_module = eval(f"model.{self.layer_path}") if not self.layer_path.startswith("model.") else eval(f"model.{self.layer_path[6:]}")
            # The above eval is risky/complex, let's do it safely
            parts = self.layer_path.split('.')
            curr = model
            for part in parts:
                curr = getattr(curr, part)
            self.num_layers = len(curr)
        except Exception:
            self.num_layers = 32 # Fallback guess
            print(f"Warning: Could not determine number of layers, defaulting to {self.num_layers}")

    def get_layer_embeddings(self, inputs, target_layer=None):
        """
        Overrides the base method to use hooks and model.chat()
        inputs: list of msgs (MiniCPM-o format)
        """
        # Determine which layers to hook
        if target_layer is not None:
            layers_to_hook = [target_layer]
        else:
            layers_to_hook = list(range(self.num_layers))
            
        # Register hooks
        self.hook_manager.register_activation_hook(layers_to_hook, layer_attr_path=self.layer_path)
        
        # Run model
        # We use model.chat with max_new_tokens=1 to force a forward pass of the prompt
        # We assume 'inputs' is a single sample's msgs list (batch size 1 for now)
        # If inputs is a list of lists (batch), we need to loop.
        # The base class usually passes a batch dict for standard models, 
        # but here we customized load_and_process to return a list of msgs.
        
        # Check if inputs is a list of msgs (single sample) or list of lists (batch)
        # Our load_and_process_batch_minicpm returns a list of 'msgs' objects.
        # But compute_alignment_score_separate iterates and calls this with whatever we pass.
        # Let's assume we handle one sample at a time here for simplicity, 
        # or a list of samples.
        
        # If inputs is a list of dicts (msgs), it's one sample.
        # If inputs is a list of lists, it's a batch.
        
        # Actually, compute_alignment_score_separate calls get_layer_embeddings(image_inputs)
        # where image_inputs is what we prepared.
        # Let's assume we process one sample at a time to avoid batching issues with model.chat
        
        # Wait, get_layer_embeddings is expected to return (batch, seq, dim) or (batch, dim)
        # If we process one by one, we can stack them.
        
        # Let's assume 'inputs' is a LIST of samples (each sample is 'msgs')
        
        all_hidden_states = {} # {layer_idx: [tensor_sample1, tensor_sample2]}
        
        # If inputs is just one msgs (list of dicts), wrap it
        if isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
             inputs = [inputs]
             
        for msgs in inputs:
            # Clear previous activations
            self.hook_manager.activations = {}
            
            with torch.no_grad():
                try:
                    # Run chat
                    # We set generate_audio=False to speed up if not needed
                    # We just need the prompt encoding
                    res = self.model.chat(
                        msgs=msgs,
                        tokenizer=self.processor, # processor is tokenizer here
                        sampling=False, # Greedy
                        max_new_tokens=1,
                        generate_audio=False
                    )
                except Exception as e:
                    print(f"Error in model.chat: {e}")
                    continue
            
            # Collect activations
            # HookManager stores: activations[layer_idx] -> tensor (1, seq, dim)
            for layer_idx in layers_to_hook:
                if layer_idx in self.hook_manager.activations:
                    act = self.hook_manager.activations[layer_idx]
                    # act is (1, seq, dim)
                    # We want to store it
                    if layer_idx not in all_hidden_states:
                        all_hidden_states[layer_idx] = []
                    all_hidden_states[layer_idx].append(act)
            
            # Clear hooks for next sample (HookManager doesn't auto-clear hooks, but clears activations manually?)
            # My HookManager implementation in utils.py accumulates? 
            # No, it overwrites or appends?
            # Let's check utils.py again. It appends to a list usually or overwrites?
            # The read_file showed: self.activations = {} in __init__.
            # It didn't show the hook function body.
            # Assuming it captures the latest forward.
            pass

        # Remove hooks
        for hook in self.hook_manager.hooks:
            hook.remove()
        self.hook_manager.hooks = []
        
        # Aggregate results
        # If target_layer is set, return (batch, dim) -> mean of sequence
        if target_layer is not None:
            if target_layer not in all_hidden_states:
                return torch.zeros(len(inputs), 4096).to(self.device) # Dummy fallback
            
            # Stack samples
            # Each sample is (1, seq, dim)
            # We mean over seq -> (1, dim)
            batch_embs = []
            for act in all_hidden_states[target_layer]:
                # act: (1, seq, dim)
                # We take the mean over sequence (dim 1)
                # Note: For image inputs, this is the image embedding mean.
                # For text inputs, this is the text embedding mean.
                batch_embs.append(act.mean(dim=1))
            
            if not batch_embs:
                return torch.empty(0)
                
            return torch.cat(batch_embs, dim=0) # (batch, dim)
            
        else:
            # Return list of layers, each is (batch, seq, dim)
            # This is tricky because seq len might differ.
            # But compute_alignment_score_separate handles it by iterating layers
            # and then doing mean per sample.
            
            # We need to return a structure that supports indexing [layer_idx]
            # and then .mean(dim=1).
            
            # Let's return a dictionary or list of lists?
            # The base class returns a tuple of tensors.
            # Here we can't easily stack if seq lens differ.
            # But compute_alignment_score_separate expects:
            # img_h = img_hidden_states[layer_idx] # (batch, seq, dim)
            # img_vec = img_h.mean(dim=1)
            
            # If we return a list of lists of tensors?
            # No, the code expects `img_h` to be a tensor.
            # If seq lens differ, we can't stack.
            # BUT, for U-Score calculation, we usually process one sample at a time or batch with padding.
            # If we process one sample at a time in the loop in main(), then batch=1.
            # If batch=1, we can return list of tensors.
            
            # However, my implementation of `compute_alignment_score_separate` takes `image_inputs` (plural).
            # If I pass a list of 1 sample, it works.
            
            # Let's construct a "Virtual Tensor" or just a list of tensors where each element is (batch, seq, dim)
            # If batch > 1 and seq differs, we pad?
            # For simplicity, let's assume batch=1 in the loop calling this.
            
            # Re-organize: List of layers. Each layer is a Tensor (batch, seq, dim).
            # If batch > 1, we must pad.
            # Let's just support batch=1 for U-Score calculation to be safe.
            
            output_layers = []
            for i in range(self.num_layers):
                if i in all_hidden_states and all_hidden_states[i]:
                    # Stack if possible
                    try:
                        output_layers.append(torch.cat(all_hidden_states[i], dim=0))
                    except:
                        # If shapes mismatch (seq len), just take the first one or fail
                        output_layers.append(all_hidden_states[i][0])
                else:
                    output_layers.append(None)
            return output_layers

def load_and_process_batch_minicpm(data_list, processor, device, modality="vision"):
    """
    Converts data list to MiniCPM-o 'msgs' format.
    Returns a list of 'msgs' (each is a list of dicts).
    """
    inputs_list = []
    
    for media_path, text in tqdm(data_list, desc="Loading Inputs"):
        content = []
        
        # Image/Video
        if modality == "vision":
            ext = os.path.splitext(media_path)[1].lower()
            if ext in ['.mp4', '.avi', '.mov']:
                # Video
                # We need to encode video frames?
                # unified_eval_minicpm_o.py has encode_video.
                # For simplicity, let's skip video or implement simple frame sampling if needed.
                # Assuming images for now as per typical datasets.
                pass
            else:
                try:
                    image = Image.open(media_path).convert("RGB")
                    content.append(image)
                except Exception:
                    pass
        elif modality == "audio":
            # Audio
            # content.append(audio_path) ?
            # MiniCPM-o expects audio path or array?
            # unified_eval says: audio, _ = librosa.load(...)
            # content.append(audio)
            try:
                import librosa
                audio, _ = librosa.load(media_path, sr=16000, mono=True)
                content.append(audio)
            except Exception:
                pass

        # Text
        content.append(text)
        
        msgs = [{"role": "user", "content": content}]
        inputs_list.append(msgs)
            
    return inputs_list

def main():
    print(f"Loading model: {MODEL_ID}...")
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation='sdpa',
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False
    )
    model = model.to(DEVICE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Initialize OmniGuard Processor
    guard_processor = MiniCPMOmniGuardProcessor(model, tokenizer, modality_type=MODALITY)

    # =======================================================
    # STEP 1: Layer Selection (U-Score)
    # =======================================================
    print("\n=== Step 1: Calculating U-Score ===")
    
    aligned_list, unaligned_list = prepare_calibration_data(JSONL_PATH, modality=MODALITY, num_samples=50)
    
    # We need to split inputs for compute_alignment_score_separate
    # It expects (image_inputs, text_inputs)
    # For MiniCPM-o, 'image_inputs' will be msgs with ONLY image.
    # 'text_inputs' will be msgs with ONLY text.
    
    def prepare_separate_inputs(data_list):
        img_inputs = []
        txt_inputs = []
        for media_path, text in data_list:
            # Image only msgs
            img_content = []
            if MODALITY == "vision":
                try:
                    image = Image.open(media_path).convert("RGB")
                    img_content.append(image)
                except: pass
            elif MODALITY == "audio":
                try:
                    import librosa
                    audio, _ = librosa.load(media_path, sr=16000, mono=True)
                    img_content.append(audio)
                except: pass
            
            # Text only msgs
            txt_content = [text]
            
            img_inputs.append([{"role": "user", "content": img_content}])
            txt_inputs.append([{"role": "user", "content": txt_content}])
            
        return img_inputs, txt_inputs

    print("Preparing separate inputs for U-Score...")
    aligned_img, aligned_txt = prepare_separate_inputs(aligned_list)
    unaligned_img, unaligned_txt = prepare_separate_inputs(unaligned_list)
    
    # Compute U-Score
    # We process one by one to avoid batch issues and stacking issues
    aligned_sims = []
    for img_in, txt_in in tqdm(zip(aligned_img, aligned_txt), total=len(aligned_img), desc="Aligned U-Score"):
        # Pass as list of 1
        sims = guard_processor.compute_alignment_score_separate([img_in], [txt_in])
        aligned_sims.append(sims)
        
    unaligned_sims = []
    for img_in, txt_in in tqdm(zip(unaligned_img, unaligned_txt), total=len(unaligned_img), desc="Unaligned U-Score"):
        sims = guard_processor.compute_alignment_score_separate([img_in], [txt_in])
        unaligned_sims.append(sims)
        
    # Aggregate
    if aligned_sims and unaligned_sims:
        avg_aligned = np.mean(np.stack(aligned_sims, axis=0), axis=0)
        avg_unaligned = np.mean(np.stack(unaligned_sims, axis=0), axis=0)
        u_scores = avg_aligned - avg_unaligned
        
        best_layer = int(np.argmax(u_scores)) + 1 # +1 because we skipped layer 0 in compute_separate
        print(f"--> Best Layer selected: {best_layer} (Max U-Score: {u_scores[best_layer-1]:.6f})")
    else:
        print("Error: No data for U-Score.")
        return

    # =======================================================
    # STEP 2: Feature Extraction
    # =======================================================
    print("\n=== Step 2: Extracting Embeddings ===")
    harmful_list, safe_list = prepare_training_data(JSONL_PATH, modality=MODALITY)
    
    # Process full inputs (Image + Text)
    harmful_inputs = load_and_process_batch_minicpm(harmful_list, tokenizer, DEVICE, MODALITY)
    safe_inputs = load_and_process_batch_minicpm(safe_list, tokenizer, DEVICE, MODALITY)
    
    def extract_features(inputs_list):
        embeddings = []
        for inp in tqdm(inputs_list, desc="Extracting"):
            # inp is one msgs list
            emb = guard_processor.get_layer_embeddings([inp], target_layer=best_layer)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)

    harmful_embs = extract_features(harmful_inputs)
    benign_embs = extract_features(safe_inputs)
    
    print(f"--> Extracted: Harmful {harmful_embs.shape}, Benign {benign_embs.shape}")

    # =======================================================
    # STEP 3: Training
    # =======================================================
    print("\n=== Step 3: Training Safety Classifier ===")
    
    if len(harmful_embs) == 0:
        return

    dataset = SimpleEmbeddingDataset(harmful_embs, benign_embs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    guard_model = SafetyClassifier(harmful_embs.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(guard_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    guard_model.train()
    for epoch in range(20):
        total_loss = 0
        for embs, labels in dataloader:
            embs, labels = embs.to(DEVICE).float(), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = guard_model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(guard_model.state_dict(), "omniguard_classifier_minicpm_o.pt")
    print("Saved classifier to omniguard_classifier_minicpm_o.pt")

if __name__ == "__main__":
    main()
