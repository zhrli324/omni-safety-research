import json
import random
import torch
import librosa
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomOmniDataset(Dataset):
    def __init__(self, jsonl_path, modality="vision", transform=None):
        """
        """
        self.data = []
        self.modality = modality
        self.transform = transform
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        
        print(f"Loaded {len(self.data)} items from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

# ================================================================
# ================================================================
def prepare_calibration_data(jsonl_path, modality="vision", num_samples=100):
    """
    """
    dataset = CustomOmniDataset(jsonl_path, modality)
    
    safe_items = [item for item in dataset.data if item.get("type") == "safe"]
    
    if len(safe_items) < 2:
        raise ValueError("安全样本太少，无法构建负样本对（至少需要2个）")
        
    if len(safe_items) > num_samples:
        safe_items = random.sample(safe_items, num_samples)
    
    aligned_data = []
    unaligned_data = []
    
    all_texts = [item["text"] for item in safe_items]
    shuffled_texts = all_texts.copy()
    random.shuffle(shuffled_texts)
    shuffled_texts = shuffled_texts[1:] + shuffled_texts[:1]

    for idx, item in enumerate(safe_items):
        media_path = item["image"] if modality == "vision" else item["audio"]
        original_text = item["text"]
        negative_text = shuffled_texts[idx]
        
        aligned_data.append((media_path, original_text))
        unaligned_data.append((media_path, negative_text))
        
    print(f"Prepared {len(aligned_data)} aligned and {len(unaligned_data)} unaligned pairs for U-Score calculation.")
    return aligned_data, unaligned_data

# ================================================================
# ================================================================
def prepare_training_data(jsonl_path, modality="vision"):
    """
    """
    dataset = CustomOmniDataset(jsonl_path, modality)
    
    harmful_data = []
    safe_data = []
    
    for item in dataset.data:
        media_path = item["image"] if modality == "vision" else item["audio"]
        text = item["text"]
        
        data_pair = (media_path, text)
        
        if item.get("type") == "harmful":
            harmful_data.append(data_pair)
        elif item.get("type") == "safe":
            safe_data.append(data_pair)
            
    print(f"Training Data Split: {len(harmful_data)} Harmful, {len(safe_data)} Safe")
    return harmful_data, safe_data

# ================================================================
# ================================================================
def load_and_process_batch(data_list, processor, device, modality="vision"):
    """
    """
    inputs_list = []
    
    for media_path, text in data_list:
        try:
            if modality == "vision":
                image = Image.open(media_path).convert("RGB")
                model_inputs = processor(text=text, images=image, return_tensors="pt")
                
            elif modality == "audio":
                audio, _ = librosa.load(media_path, sr=16000) 
                model_inputs = processor(text=text, audios=audio, return_tensors="pt", sampling_rate=16000)
            
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            inputs_list.append(model_inputs)
            
        except Exception as e:
            print(f"Error processing {media_path}: {e}")
            continue
            
    return inputs_list