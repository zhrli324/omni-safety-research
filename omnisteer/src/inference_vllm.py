import os
import argparse
import torch
from tqdm import tqdm
from utils import load_model, SafetyDataset
from modeling import SafetyAdapter
from PIL import Image
from qwen_omni_utils import process_mm_info
import json
import numpy as np
import librosa
import soundfile as sf
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode

# Try to import vllm and vllm_omni
# try:
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni
# except ImportError:
#     print("Error: vllm or vllm_omni is not installed. Please install them to use this script.")
#     exit(1)

SEED = 42

def apply_safety_adapter(omni_instance, adapter_path, device="cuda"):
    """
    Loads the SafetyAdapter and merges its bias into the model's embedding weights.
    This attempts to locate the 'thinker' model within the Omni pipeline.
    """
    print(f"Loading SafetyAdapter from {adapter_path}...")
    
    # Load adapter weights
    # We need hidden_size. 
    # We'll try to infer it from the adapter file or default to Qwen2.5-Omni size (usually 3584 for 7B)
    # But better to get it from the model if possible.
    
    # Introspection to find the model
    # Omni structure is likely: omni_instance -> thinker (LLM) -> ...
    # or omni_instance.engine ...
    
    model = None
    # Attempt to find the underlying vLLM model in the Omni instance
    # This is heuristic as we don't have the source code of vllm_omni
    potential_attrs = ["thinker", "llm", "model", "engine"]
    
    for attr in potential_attrs:
        if hasattr(omni_instance, attr):
            obj = getattr(omni_instance, attr)
            print(f"Found attribute '{attr}' in Omni instance.")
            # Check if this object looks like a vLLM engine or model wrapper
            if hasattr(obj, "llm_engine"):
                # It's likely a vLLM LLM object
                model_executor = obj.llm_engine.model_executor
                driver_worker = model_executor.driver_worker
                if hasattr(driver_worker, "model_runner"):
                    model = driver_worker.model_runner.model
                elif hasattr(driver_worker, "model"):
                    model = driver_worker.model
                break
            elif hasattr(obj, "model"):
                 # Maybe it's a direct wrapper
                 model = obj.model
                 break
    
    if model is None:
        print("Warning: Could not locate the underlying PyTorch model in the Omni instance. SafetyAdapter injection skipped.")
        return

    print("Located underlying model. Proceeding with injection.")
    
    # Find embedding layer
    embed_layer = None
    if hasattr(model, "embed_tokens"): 
        embed_layer = model.embed_tokens
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_layer = model.model.embed_tokens
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        embed_layer = model.transformer.wte
    
    if embed_layer is None:
        for name, module in model.named_modules():
            if "embed_tokens" in name or "wte" in name or isinstance(module, torch.nn.Embedding):
                embed_layer = module
                print(f"Found embedding layer at: {name}")
                break
    
    if embed_layer is None:
        print("Error: Could not find embedding layer. Injection failed.")
        return

    # Determine hidden size from the model
    hidden_size = embed_layer.weight.shape[1]
    print(f"Model hidden size: {hidden_size}")

    adapter = SafetyAdapter(hidden_size)
    try:
        adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
    except Exception as e:
        print(f"Error loading adapter state dict: {e}")
        return

    safety_bias = adapter.bias.detach().squeeze()
    safety_bias = safety_bias.to(embed_layer.weight.device).to(embed_layer.weight.dtype)
    
    with torch.no_grad():
        embed_layer.weight.add_(safety_bias)
        
    print("SafetyAdapter injected successfully into Omni model.")

def construct_omni_prompt(item, default_system):
    """
    Constructs the prompt and multi_modal_data for vllm-omni.
    """
    # Extract content
    text_query = item.get("text", "")
    image_path = item.get("image")
    video_path = item.get("video")
    audio_path = item.get("audio")
    
    # Build prompt string with special tokens
    # Based on vllm-omni example
    
    prompt_parts = [f"<|im_start|>system\n{default_system}<|im_end|>\n", "<|im_start|>user\n"]
    
    multi_modal_data = {}
    limit_mm = {}
    
    if audio_path:
        prompt_parts.append("<|audio_bos|><|AUDIO|><|audio_eos|>")
        if os.path.exists(audio_path):
            audio_signal, sr = librosa.load(audio_path, sr=16000)
            multi_modal_data["audio"] = (audio_signal.astype(np.float32), sr)
            limit_mm["audio"] = 1
            
    if image_path:
        prompt_parts.append("<|vision_bos|><|IMAGE|><|vision_eos|>")
        if os.path.exists(image_path):
            pil_image = Image.open(image_path)
            multi_modal_data["image"] = convert_image_mode(pil_image, "RGB")
            limit_mm["image"] = 1

    if video_path:
        prompt_parts.append("<|vision_bos|><|VIDEO|><|vision_eos|>")
        if os.path.exists(video_path):
            # Default num_frames=16 from example
            multi_modal_data["video"] = video_to_ndarrays(video_path, num_frames=16)
            limit_mm["video"] = 1

    prompt_parts.append(f"{text_query}<|im_end|>\n")
    prompt_parts.append("<|im_start|>assistant\n")
    
    prompt = "".join(prompt_parts)
    
    return {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data
    }, limit_mm

def inference(args):
    # Initialize Omni
    print(f"Initializing vllm-omni with model: {args.model_path}")
    
    # Omni parameters from example
    omni_llm = Omni(
        model=args.model_path,
        log_stats=False,
        init_sleep_seconds=20, # Allow initialization
        batch_timeout=5,
        init_timeout=300,
        shm_threshold_bytes=65536,
        # trust_remote_code might be needed if passed to underlying vllm, 
        # but Omni constructor signature in example doesn't show it.
        # Assuming it handles it or we can't pass it.
    )
    
    # Apply Safety Adapter
    # Note: This might fail if Omni spawns isolated processes we can't reach.
    apply_safety_adapter(omni_llm, args.adapter_path)
    
    # Prepare Sampling Params
    # From example
    thinker_sampling_params = SamplingParams(
        temperature=0.0, 
        top_p=1.0, 
        top_k=-1, 
        max_tokens=2048,
        seed=SEED, 
        detokenize=True,
        repetition_penalty=1.1,
    )
    
    # Only use thinker params for text-only output, skipping talker and code2wav
    sampling_params_list = [
        thinker_sampling_params,
    ]

    # Load Dataset
    # We need a processor just to load the dataset if SafetyDataset depends on it, 
    # but SafetyDataset only uses processor for apply_chat_template which we might not use here 
    # if we construct prompts manually.
    # However, SafetyDataset constructor requires it.
    from transformers import AutoProcessor
    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    except:
        processor = None # Fallback if not needed for data loading itself

    dataset = SafetyDataset(args.data_path, processor)
    
    print("Preparing inputs...")
    default_system = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
        "Group, capable of perceiving auditory and visual inputs, as well as "
        "generating text and speech."
    )
    
    prompts = []
    ids = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        inputs, _ = construct_omni_prompt(item, default_system)
        prompts.append(inputs)
        ids.append(item.get("id", i))
        
    print(f"Starting generation for {len(prompts)} items...")
    
    # Run generation
    # Omni.generate takes a list of prompts
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)
    
    # Process results
    results = []
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # omni_outputs is a list of StageOutput (one per stage? or one per batch?)
    # The example iterates: for stage_outputs in omni_outputs:
    
    # We need to map back to our IDs. 
    # The example uses request_id which seems to be the index in the prompts list.
    
    with open(args.output_file, 'w') as f:
        for stage_outputs in omni_outputs:
            # We are interested in the text output from the "thinker" or final output if it's text
            # The example checks final_output_type
            
            if stage_outputs.final_output_type == "text":
                for output in stage_outputs.request_output:
                    req_id = int(output.request_id)
                    text_output = output.outputs[0].text
                    
                    # Clean up
                    if "assistant\n" in text_output:
                        response = text_output.split("assistant\n")[-1].strip()
                    else:
                        response = text_output.strip()
                        
                    result_entry = {
                        "id": ids[req_id],
                        "response": response
                    }
                    f.write(json.dumps(result_entry) + "\n")
                    results.append(result_entry)
            
            # If we want audio, we can handle it here too, but the user asked for inference logic similar to original
            # which saved text responses.
            
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="output/inference_results_vllm_omni.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    inference(args)
