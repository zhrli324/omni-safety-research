import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from utils import load_model, SafetyDataset
from modeling import SafetyAdapter, SafetyModelWrapper
from PIL import Image
from qwen_omni_utils import process_mm_info

def train(args):
    # 1. Load Model
    model, processor = load_model(args.model_path, args.device)
    # Freeze original model
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Initialize Adapter
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 3584 # Default for Qwen2.5-7B
    adapter = SafetyAdapter(hidden_size).to(args.device)
    wrapper = SafetyModelWrapper(model, adapter)
    
    # 3. Load Refusal Vector
    refusal_vector = torch.load(args.vector_path).to(args.device)
    refusal_vector = refusal_vector / torch.norm(refusal_vector) # Ensure normalized
    
    # 4. Setup Optimizer
    optimizer = AdamW(adapter.parameters(), lr=args.lr)
    
    # 5. Setup Hook for Loss Calculation (Get activation at target layer)
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            activations[name] = hidden_state
        return hook
        
    # Register hook on the target layer to monitor activations
    layer = model.thinker.model.layers[args.layer_idx]
    layer.register_forward_hook(get_activation("target_layer"))
    
    # 6. Training Loop
    dataset = SafetyDataset(args.data_path, processor)
    # Simple data loader (batch size 1 for simplicity with multimodal)
    # In production, use a proper DataLoader with collate_fn
    
    print("Starting training...")
    model.train() # Set to train mode (though model is frozen, adapter is not)
    # Note: If model has BatchNorm/Dropout, might want to keep it in eval mode.
    # Usually LLMs don't have BN. Dropout should probably be off.
    model.eval() 
    adapter.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            
            # Prepare inputs
            content = []
            if "video" in item: content.append({"type": "video", "video": item["video"]})
            if "image" in item: content.append({"type": "image", "image": Image.open(item["image"]).convert("RGB")})
            if "audio" in item: content.append({"type": "audio", "audio": item["audio"]})
            if "text" in item: content.append({"type": "text", "text": item["text"]})
            
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
                {"role": "user", "content": content}
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            
            optimizer.zero_grad()
            
            # Forward pass
            # The wrapper handles the injection
            model.thinker(**inputs)
            
            # Get activation
            act = activations["target_layer"] # [1, seq_len, dim]
            last_token_act = act[:, -1, :] # [1, dim]
            
            # Calculate Projection
            # proj = dot(h, v)
            # Ensure refusal_vector is 1D and last_token_act is 1D for dot product
            # If last_token_act is [1, dim], squeeze makes it [dim]
            # If refusal_vector is [dim] or [1, dim], we need to handle it.
            
            vec = refusal_vector.squeeze()
            token_act = last_token_act.squeeze()
            
            # Ensure both are 1D
            if vec.dim() > 1:
                vec = vec.view(-1)
            if token_act.dim() > 1:
                token_act = token_act.view(-1)
                
            projection = torch.dot(token_act, vec)
            
            # Calculate Loss
            # If Harmful: We want Projection > Threshold (Align with refusal)
            # Loss = ReLU(Threshold_High - Projection)
            # If Safe: We want Projection < Threshold (Away from refusal)
            # Loss = ReLU(Projection - Threshold_Low)
            
            loss = 0
            if item["type"] == "harmful":
                loss = torch.relu(args.threshold_high - projection)
            else:
                loss = torch.relu(projection - args.threshold_low)
                
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(dataset)}")
        
    # Save Adapter
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    wrapper.save_adapter(args.output_path)
    print(f"Adapter saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/safety_adapter.pt")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--threshold_high", type=float, default=0.5, help="Target projection for harmful")
    parser.add_argument("--threshold_low", type=float, default=-0.5, help="Target projection for safe")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
