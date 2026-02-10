import os
import argparse
import torch
from tqdm import tqdm
from utils import load_model, SafetyDataset
from modeling import SafetyAdapter, SafetyModelWrapper
from PIL import Image
from qwen_omni_utils import process_mm_info
import json

def inference(args):
    model, processor = load_model(args.model_path, args.device)
    
    # Load Adapter
    hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else 3584
    # adapter = SafetyAdapter(hidden_size).to(args.device)
    # wrapper = SafetyModelWrapper(model, adapter)
    # wrapper.load_adapter(args.adapter_path)
    
    dataset = SafetyDataset(args.data_path, processor)
    
    print("Starting inference...")
    results = []
    
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        
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
        
        with torch.no_grad():
            # Generate
            # We need to call the generate method of the underlying model
            # But we need the hook to be active.
            # The wrapper.forward calls model(), but generate() is different.
            # However, generate() eventually calls model() (forward).
            # So wrapper.model.generate() should work IF the hook is on wrapper.model.thinker.model
            
            # Note: wrapper.model is the Qwen2_5OmniForConditionalGeneration

            gen_kwargs = {
                "max_new_tokens": 400,
                "do_sample": True,
                "temperature": 0.7,
            }
            # generated_ids = wrapper.model.generate(**inputs, **gen_kwargs)
            generated_ids = model.generate(**inputs, **gen_kwargs)
            
            # Handle tuple output from generate (common in some configurations)
            if isinstance(generated_ids, tuple):
                generated_ids = generated_ids[0]

            # Decode response
            response_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # Extract assistant response if needed (similar to image-audio-text.py)
            if "assistant\n" in response_text:
                response = response_text.split("assistant\n")[-1].strip()
            else:
                response = response_text

            with open(args.output_file, 'a') as f:
                f.write(json.dumps({
                    "id": item.get("id", i),
                    "type": item.get("type", "unknown"),
                    "response": response
                }) + "\n")
            
        results.append({
            "id": item.get("id", i),
            "type": item.get("type", "unknown"),
            "response": response
        })
        
    # Save results
    # with open(args.output_file, 'w') as f:
    #     for res in results:
    #         f.write(json.dumps(res) + "\n")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="output/inference_results.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    inference(args)
