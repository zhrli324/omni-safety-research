import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

def load_model(model_path, device="cuda"):
    """
    Loads the Qwen2.5-Omni model and processor.
    """
    # Note: Adjust trust_remote_code=True if necessary for custom models

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    return model, processor

class SafetyDataset(Dataset):
    def __init__(self, data_path, processor):
        self.data = []
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        elif data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported data format. Use .json or .jsonl files.")
        self.processor = processor
        # self.data = self.data[:300]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

def collate_fn(batch, processor, device="cuda"):
    """
    Custom collate function to process multimodal inputs.
    """
    # This needs to be adapted to the specific input format of Qwen2.5-Omni
    # Assuming we use the chat template logic
    
    texts = []
    audios = []
    images = []
    videos = []
    
    labels = [] # 'harmful' or 'safe'
    
    for item in batch:
        # Construct conversation
        content = []
        if "image" in item:
            content.append({"type": "image", "image": item["image"]})
        if "audio" in item:
            content.append({"type": "audio", "audio": item["audio"]})
        if "video" in item:
            content.append({"type": "video", "video": item["video"]})
        if "text" in item:
            content.append({"type": "text", "text": item["text"]})
            
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": content}
        ]
        
        # Apply template
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        texts.append(text)
        
        # Collect media paths (Logic depends on how processor handles lists)
        # Here we assume the processor can handle lists of paths if we structure them right
        # Or we might need to load them. 
        # For simplicity, we assume the processor handles the raw paths/inputs if passed correctly.
        # NOTE: In the user's provided code, they had a `process_mm_info` function. 
        # We should probably try to import that or replicate it if we had it.
        # Since we don't have the full `qwen_omni_utils`, we will assume standard processor usage
        # or placeholders.
        
        # Placeholder for media loading logic
        # In a real implementation, you would load images/audio here.
        
        labels.append(1 if item["type"] == "harmful" else 0)

    # Tokenize
    # Note: This is a simplified call. Real Qwen-Omni might need specific audio/image args.
    # We will assume the user has the environment set up where processor handles this.
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    
    return {k: v.to(device) for k, v in inputs.items()}, torch.tensor(labels).to(device)

def calculate_refusal_vector(safe_acts, harmful_acts):
    """
    Calculates the direction vector from safe to harmful (or vice versa).
    Here: Harmful - Safe (Direction towards refusal/harmful)
    Wait, usually Refusal Vector is (Refusal State - Non-Refusal State).
    If we want to detect harmful queries, we might look at the activation difference.
    
    Standard approach: Mean(Harmful_Activations) - Mean(Safe_Activations)
    """
    safe_mean = torch.mean(torch.stack(safe_acts), dim=0)
    harm_mean = torch.mean(torch.stack(harmful_acts), dim=0)
    refusal_vector = harm_mean - safe_mean
    return refusal_vector / torch.norm(refusal_vector)
