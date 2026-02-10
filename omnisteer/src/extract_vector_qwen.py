import os
import argparse
import torch
import json
from tqdm import tqdm
from utils import load_model, calculate_refusal_vector, SafetyDataset

def extract_vector(args):
    model, processor = load_model(args.model_path, args.device)
    model.eval()
    
    # Hook to capture activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            # output is usually a tuple (hidden_states, ...)
            # We take the last token's hidden state
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            activations[name] = hidden_state.detach()
        return hook

    # Register hook on the target layer
    # Path: model.thinker.model.layers[layer_idx]
    layer = model.thinker.model.layers[args.layer_idx]
    layer.register_forward_hook(get_activation(f"layer_{args.layer_idx}"))
    
    dataset = SafetyDataset(args.data_path, processor)
    
    safe_acts = []
    harmful_acts = []
    
    print("Extracting activations...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
        # Construct inputs (Simplified, reuse logic from utils or inline)
        # For extraction, we process one by one
        content = []
        if "prompt" in item: content.append({"type": "text", "text": item["prompt"]})
        # Add other modalities if present in dataset
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": content}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt")
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
            
        act = activations[f"layer_{args.layer_idx}"] # [1, seq_len, hidden_dim]
        last_token_act = act[:, -1, :].cpu() # [1, hidden_dim]
        
        if item["type"] == "harmful":
            harmful_acts.append(last_token_act)
        else:
            safe_acts.append(last_token_act)
            
    print(f"Collected {len(safe_acts)} safe and {len(harmful_acts)} harmful activations.")
    
    if not safe_acts or not harmful_acts:
        print("Error: Need both safe and harmful samples.")
        return

    refusal_vector = calculate_refusal_vector(safe_acts, harmful_acts)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(refusal_vector, args.output_path)
    print(f"Refusal vector saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/refusal_vector.pt")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    extract_vector(args)
