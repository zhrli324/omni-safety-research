import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from PIL import Image
from omniguard_core import OmniGuardProcessor, SafetyClassifier, SimpleEmbeddingDataset
from custom_data_utils import prepare_calibration_data, prepare_training_data
from qwen_omni_utils import process_mm_info
import os
from tqdm import tqdm

# ============================
# ============================
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MODALITY = "vision" # "vision" or "audio"
JSONL_PATH = "data/holisafe_abs.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class QwenOmniGuardProcessor(OmniGuardProcessor):
    def _get_modality_masks(self, input_ids):
        """
        Qwen-Omni specific modality mask generation.
        """
        # Get special token IDs
        # Note: These might vary based on exact model config, retrieving from tokenizer/config is best
        # Assuming standard Qwen-Omni tokens
        
        # We can iterate through input_ids and find start/end tokens
        # But for efficiency, let's try to identify ranges.
        
        # Qwen-Omni uses <|video_start|> ... <|video_end|> etc.
        # The tokens between these are the modality tokens.
        
        # Let's assume we can get these IDs from the processor/tokenizer
        tokenizer = self.processor.tokenizer
        
        # Define start/end tokens for different modalities
        # Adjust based on actual tokenizer vocab
        
        modality_tokens = {
            "vision": [
                ("<|image_start|>", "<|image_end|>"), 
                ("<|video_start|>", "<|video_end|>"),
                ("<|vision_start|>", "<|vision_end|>"), # Add Qwen2-VL style tokens
                ("<|vision_bos|>", "<|vision_eos|>")
            ],
            "audio": [("<|audio_start|>", "<|audio_end|>")]
        }
        
        target_pairs = modality_tokens.get(self.modality_type, [])
        if self.modality_type == "vision":
             # Include both image and video for vision modality
             target_pairs = modality_tokens["vision"]
        
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Strategy 1: Check for start/end token pairs
        for start_token, end_token in target_pairs:
            if start_token in tokenizer.get_vocab() and end_token in tokenizer.get_vocab():
                start_id = tokenizer.convert_tokens_to_ids(start_token)
                end_id = tokenizer.convert_tokens_to_ids(end_token)
                
                for i in range(input_ids.shape[0]):
                    row = input_ids[i]
                    start_indices = (row == start_id).nonzero(as_tuple=True)[0]
                    end_indices = (row == end_id).nonzero(as_tuple=True)[0]
                    
                    for s_idx, e_idx in zip(start_indices, end_indices):
                        vision_mask[i, s_idx+1:e_idx] = True
                        
        # Strategy 2: Check for specific image pad tokens (common in Qwen2-VL)
        if vision_mask.sum() == 0:
            # Try finding <|image_pad|> or similar
            possible_pad_tokens = ["<|image_pad|>", "<|video_pad|>", "<|IMAGE|>"]
            for pad_token in possible_pad_tokens:
                # Check if token exists in tokenizer
                # Using convert_tokens_to_ids is safer than get_vocab() for special tokens
                pad_id = tokenizer.convert_tokens_to_ids(pad_token)
                if pad_id is not None and pad_id != tokenizer.unk_token_id:
                    vision_mask |= (input_ids == pad_id)
        
        if vision_mask.sum() == 0:
             print("DEBUG: No vision tokens found in mask! Checked pairs and pad tokens.")
        else:
             # print(f"DEBUG: Found {vision_mask.sum()} vision tokens.")
             pass
        
        text_mask = ~vision_mask
        return vision_mask, text_mask

def load_and_process_batch_qwen(data_list, processor, device, modality="vision"):
    """
    Qwen-Omni specific batch processing using process_mm_info.
    """
    inputs_list = []
    
    for media_path, text in tqdm(data_list, desc="Loading Inputs"):
        try:
            # Construct conversation format expected by process_mm_info
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            user_content = conversation[1]["content"]
            
            if modality == "vision":
                # Determine if image or video based on extension
                ext = os.path.splitext(media_path)[1].lower()
                if ext in ['.mp4', '.avi', '.mov']:
                    user_content.append({"type": "video", "video": media_path})
                else:
                    # Assume image
                    # process_mm_info expects PIL image for 'image' type usually, or path?
                    # unified_eval_qwen.py loads PIL image.
                    image = Image.open(media_path).convert("RGB")
                    user_content.append({"type": "image", "image": image})
            elif modality == "audio":
                user_content.append({"type": "audio", "audio": media_path})
                
            user_content.append({"type": "text", "text": text})
            
            # Apply template
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Process MM info
            # process_mm_info returns (audios, images, videos)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            # Processor call
            model_inputs = processor(
                text=text_prompt,
                images=images,
                videos=videos,
                audios=audios,
                padding=True,
                return_tensors="pt",
            )
            
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            inputs_list.append(model_inputs)
            
        except Exception as e:
            print(f"Error processing {media_path}: {e}")
            continue
            
    return inputs_list

def main():
    print(f"Loading model: {MODEL_ID}...")
    # processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype="auto")
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)

    guard_processor = QwenOmniGuardProcessor(model, processor, modality_type=MODALITY)

    # =======================================================
    # =======================================================
    print("\n=== Step 1: Calculating U-Score ===")
    
    print("Step1: Preparing calibration data...")
    aligned_list, unaligned_list = prepare_calibration_data(JSONL_PATH, modality=MODALITY, num_samples=50)
    print(f"  -> Prepared calibration lists: aligned={len(aligned_list)}, unaligned={len(unaligned_list)}")

    aligned_sims = []
    unaligned_sims = []

    print("\nStep2: Processing aligned samples...")
    aligned_inputs = load_and_process_batch_qwen(aligned_list, processor, DEVICE, modality=MODALITY)
    print(f"  -> Loaded {len(aligned_inputs)} aligned input(s) into model format.")
    
    # DEBUG: Print tokens for the first sample
    if len(aligned_inputs) > 0:
        debug_ids = aligned_inputs[0]["input_ids"][0]
        print(f"DEBUG: First sample input_ids length: {len(debug_ids)}")
        # Decode and print a snippet
        decoded = processor.tokenizer.decode(debug_ids, skip_special_tokens=False)
        print(f"DEBUG: Decoded first sample (truncated): {decoded[:500]} ...")
        # Print special tokens in the sequence
        special_ids = [id.item() for id in debug_ids if id.item() >= processor.tokenizer.vocab_size or id.item() in processor.tokenizer.all_special_ids]
        print(f"DEBUG: Special token IDs found: {special_ids[:20]} ...")
        
    for idx, inputs in enumerate(aligned_inputs, start=1):
        try:
            print(f"    Processing aligned sample {idx}/{len(aligned_inputs)} - input keys: {list(inputs.keys())}")
            # show tensor shapes for quick debugging
            for k, v in inputs.items():
                try:
                    print(f"      {k}: {tuple(v.shape)}")
                except Exception:
                    print(f"      {k}: (shape unknown)")
            sims = guard_processor.compute_alignment_score(inputs)
            # normalize to numpy for consistent aggregation & printing
            try:
                sims_arr = sims.detach().cpu().numpy()
            except Exception:
                sims_arr = np.array(sims)
            print(f"      sims shape: {sims_arr.shape}, sample values (first 5): {sims_arr.flatten()[:5]}")
            aligned_sims.append(sims_arr)
        except Exception as e:
            print(f"    Error computing aligned sample {idx}: {e}")
            continue

    print("\nStep3: Processing unaligned samples...")
    unaligned_inputs = load_and_process_batch_qwen(unaligned_list, processor, DEVICE, modality=MODALITY)
    print(f"  -> Loaded {len(unaligned_inputs)} unaligned input(s) into model format.")
    for idx, inputs in enumerate(unaligned_inputs, start=1):
        try:
            print(f"    Processing unaligned sample {idx}/{len(unaligned_inputs)} - input keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                try:
                    print(f"      {k}: {tuple(v.shape)}")
                except Exception:
                    print(f"      {k}: (shape unknown)")
            sims = guard_processor.compute_alignment_score(inputs)
            try:
                sims_arr = sims.detach().cpu().numpy()
            except Exception:
                sims_arr = np.array(sims)
            print(f"      sims shape: {sims_arr.shape}, sample values (first 5): {sims_arr.flatten()[:5]}")
            unaligned_sims.append(sims_arr)
        except Exception as e:
            print(f"    Error computing unaligned sample {idx}: {e}")
            continue

    # [Configuration/Note]
    print("\nStep4: Calculating U-Scores...")
    if aligned_sims and unaligned_sims:
        try:
            avg_aligned = np.mean(np.stack(aligned_sims, axis=0), axis=0)
            avg_unaligned = np.mean(np.stack(unaligned_sims, axis=0), axis=0)
            u_scores = avg_aligned - avg_unaligned
            print(f"  -> avg_aligned shape: {avg_aligned.shape}, avg_unaligned shape: {avg_unaligned.shape}")
            # show top few layers with highest U
            ranked = np.argsort(-u_scores)
            top_k = min(5, len(u_scores))
            print(f"  -> Top {top_k} layers by U-Score:")
            for i in range(top_k):
                layer = ranked[i]
                print(f"     Layer {layer}: U-Score={u_scores[layer]:.6f}")
            best_layer = int(np.argmax(u_scores))
            print(f"--> Best Layer selected: {best_layer} (Max U-Score: {u_scores[best_layer]:.6f})")
        except Exception as e:
            print(f"Error during U-Score aggregation: {e}")
            return
    else:
        print(f"Error: No valid data for U-Score calculation. aligned_sims={len(aligned_sims)}, unaligned_sims={len(unaligned_sims)}")
        return

    # =======================================================
    # =======================================================
    print("\n=== Step 2: Extracting Embeddings ===")
    
    harmful_list, safe_list = prepare_training_data(JSONL_PATH, modality=MODALITY)
    
    def extract_features_from_list(data_list):
        processed_inputs = load_and_process_batch_qwen(data_list, processor, DEVICE, modality=MODALITY)
        embeddings = []
        for inputs in tqdm(processed_inputs, desc="Extracting Features"):
            emb = guard_processor.get_layer_embeddings(inputs, target_layer=best_layer)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)

    harmful_embs = extract_features_from_list(harmful_list)
    benign_embs = extract_features_from_list(safe_list)
    
    print(f"--> Extracted: Harmful {harmful_embs.shape}, Benign {benign_embs.shape}")
    
    # =======================================================
    # =======================================================
    print("\n=== Step 3: Training Safety Classifier ===")
    
    if len(harmful_embs) == 0 or len(benign_embs) == 0:
        print("Not enough data to train. Exiting demo.")
        return

    dataset = SimpleEmbeddingDataset(harmful_embs, benign_embs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    input_dim = harmful_embs.shape[1]
    guard_model = SafetyClassifier(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(guard_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    guard_model.train()
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0
        for embs, labels in dataloader:
            # Ensure embs are float32 to match the model weights
            embs, labels = embs.to(DEVICE).float(), labels.to(DEVICE).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = guard_model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    print("\n--> Training Complete! Guard Model is ready.")
    
    # Save the model
    torch.save(guard_model.state_dict(), "omniguard_classifier_qwen.pt")
    print("Saved classifier to omniguard_classifier_qwen.pt")

if __name__ == "__main__":
    main()