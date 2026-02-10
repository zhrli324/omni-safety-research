import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image

# ==========================================
# ==========================================
class SafetyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256):
        super(SafetyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ==========================================
# ==========================================
class OmniGuardProcessor:
    def __init__(self, model, processor, modality_type="vision"):
        """
        """
        self.model = model
        self.processor = processor
        self.modality_type = modality_type
        self.device = model.device

    def _get_modality_masks(self, input_ids):
        """
        """
        seq_len = input_ids.shape[1]
        
        if self.modality_type == "vision":
            vocab_size = self.processor.tokenizer.vocab_size
            vision_mask = input_ids >= vocab_size
            # [Configuration/Note]
            
        elif self.modality_type == "audio":
            vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            # start_idx = ...
            # end_idx = ...
            # vision_mask[:, start_idx:end_idx] = True
            pass 
        
        if 'vision_mask' not in locals():
            raise NotImplementedError("请在 _get_modality_masks 中实现你模型的 Token 分离逻辑")

        text_mask = ~vision_mask
        return vision_mask, text_mask

    def get_layer_embeddings(self, inputs, target_layer=None):
        """
        """
        with torch.no_grad():
            # [Configuration/Note]
            # Handle different model structures
            if hasattr(self.model, "thinker"):
                outputs = self.model.thinker(**inputs, output_hidden_states=True, use_cache=False)
            elif hasattr(self.model, "model"):
                # Some models like Baichuan/Llama wrap the core model in .model
                # But usually .forward() on the wrapper handles it.
                # However, we need output_hidden_states.
                # If we call self.model(**inputs), it should work if it supports the arg.
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
            else:
                # Fallback
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        
        hidden_states = outputs.hidden_states # Tuple of (batch, seq, dim)
        
        if target_layer is not None:
            # hidden_states[target_layer]: (batch, seq, dim)
            # mean(1) -> (batch, dim)
            return hidden_states[target_layer].mean(dim=1)
        
        return hidden_states

    def compute_alignment_score_separate(self, image_inputs, text_inputs):
        """
        Compute alignment score using separate inputs for Image and Text.
        This avoids the need for complex masking in mixed sequences.
        """
        img_hidden_states = self.get_layer_embeddings(image_inputs)
        txt_hidden_states = self.get_layer_embeddings(text_inputs)
        
        sims = []
        # Start from layer 1
        # Ensure we don't go out of bounds if models have different depths (should be same model though)
        num_layers = min(len(img_hidden_states), len(txt_hidden_states))
        
        for layer_idx in range(1, num_layers):
            img_h = img_hidden_states[layer_idx] # (batch, seq_img, dim)
            txt_h = txt_hidden_states[layer_idx] # (batch, seq_txt, dim)
            
            # Average pooling
            img_vec = img_h.mean(dim=1) # (batch, dim)
            txt_vec = txt_h.mean(dim=1) # (batch, dim)
            
            # Cosine Similarity
            cos_sim = F.cosine_similarity(img_vec, txt_vec, dim=-1).mean().item()
            sims.append(cos_sim)
            
        return np.array(sims)

    def compute_alignment_score(self, inputs):
        """
        """
        hidden_states = self.get_layer_embeddings(inputs)
        input_ids = inputs["input_ids"]
        
        vision_mask, text_mask = self._get_modality_masks(input_ids)
        
        sims = []
        for layer_idx, layer_h in enumerate(hidden_states[1:]):
            # layer_h: (batch, seq, dim)
            
            # Ensure masks are on the same device as the current layer hidden state
            # This is crucial when using device_map="auto" which might spread layers across GPUs
            current_device = layer_h.device
            if vision_mask.device != current_device:
                vision_mask = vision_mask.to(current_device)
            if text_mask.device != current_device:
                text_mask = text_mask.to(current_device)
            
            # [Configuration/Note]
            if layer_h.shape[0] == 1:
                # Check if masks are empty to avoid NaN
                if vision_mask[0].sum() == 0 or text_mask[0].sum() == 0:
                    sims.append(0.0) # Return 0 similarity if modality is missing
                    continue
                    
                vis_h = layer_h[:, vision_mask[0], :].mean(1) # (1, dim)
                text_h = layer_h[:, text_mask[0], :].mean(1)  # (1, dim)
            else:
                vis_sum = vision_mask.sum(1, keepdim=True)
                text_sum = text_mask.sum(1, keepdim=True)
                
                # Avoid division by zero
                vis_sum = torch.clamp(vis_sum, min=1e-9)
                text_sum = torch.clamp(text_sum, min=1e-9)
                
                vis_h = layer_h * vision_mask.unsqueeze(-1)
                vis_h = vis_h.sum(1) / vis_sum
                
                text_h = layer_h * text_mask.unsqueeze(-1)
                text_h = text_h.sum(1) / text_sum

            cos_sim = F.cosine_similarity(vis_h, text_h, dim=-1).mean().item()
            sims.append(cos_sim)
            
        return np.array(sims)

# ==========================================
# ==========================================
class SimpleEmbeddingDataset(Dataset):
    def __init__(self, harmful_embeddings, benign_embeddings):
        """
        :param harmful_embeddings: Tensor (N, dim)
        :param benign_embeddings: Tensor (M, dim)
        """
        self.data = torch.cat([harmful_embeddings, benign_embeddings], dim=0)
        # Label: 1 for harmful, 0 for benign
        self.labels = torch.cat([
            torch.ones(len(harmful_embeddings)), 
            torch.zeros(len(benign_embeddings))
        ], dim=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]