import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from utils import load_model, HookManager, calculate_refusal_vector
from qwen_omni_utils import process_mm_info
import torch.nn.functional as F

def run_vector_similarity_experiment(args):
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device)
    
    hook_manager = HookManager(model)
    target_layers = list(range(15, 25))
    layer_path = "thinker.model.layers" 
    hook_manager.register_activation_hook(target_layers, layer_attr_path=layer_path)
    
    with open(args.data_file, 'r') as f:
        data = json.load(f)
        
    # ==========================================
    # ==========================================
    print("\nStep 1: Calculating Text Refusal Vector (V_text)...")
    text_activations = {"safe": [], "harmful": []}
    
    # train_subset = data["text_train"][:100] if len(data["text_train"]) > 100 else data["text_train"]
    train_subset = data["text_train"]
    
    for item in tqdm(train_subset, desc="Processing Text Data"):
        content = [{"type": "text", "text": item["text"]}]
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": content}
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
            
        for layer_idx in target_layers:
            act = hook_manager.activations[layer_idx][:, -1, :]
            if layer_idx not in text_activations:
                text_activations[layer_idx] = {"safe": [], "harmful": []}
            text_activations[layer_idx][item["type"]].append(act)
        hook_manager.activations = {}

    v_text = {}
    for layer_idx in target_layers:
        safe_acts = text_activations[layer_idx]["safe"]
        harm_acts = text_activations[layer_idx]["harmful"]
        if safe_acts and harm_acts:
            v_text[layer_idx] = calculate_refusal_vector(safe_acts, harm_acts)

    # ==========================================
    # ==========================================
    print("\nStep 2: Calculating Multi-Modal Refusal Vector (V_mm)...")
    
    # mm_activations[modality][layer_idx][label] = [act, ...]
    mm_activations = {} 
    
    target_modalities = ["image", "audio", "video"] 
    
    max_samples_per_type = 200
    counts = {m: {"safe": 0, "harmful": 0} for m in target_modalities}
    
    for item in tqdm(data["cross_modal_test"], desc="Processing Cross-Modal Data"):
        if item["modality"] == "image" and item["label"] == "harmful" and int(item["id"]) not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 303, 304, 305, 306, 307, 308, 310, 311, 312, 313, 314, 315, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 332, 333, 334, 335, 336, 339, 340, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 353, 354, 356, 357, 358, 360, 361, 362, 363, 364, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 378, 379, 380, 381, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 488, 490, 491, 492, 493, 494, 495, 496, 497, 498, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 514, 516, 517, 518, 519]:
            continue
        if item["modality"] == "audio" and item["label"] == "harmful" and int(item["id"]) not in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 88, 89, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 108, 109, 110, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 141, 142, 143, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 182, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209, 213, 214, 215, 216, 217, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 244, 245, 246, 247, 248, 249, 250, 252, 253, 255, 256, 259, 260, 262, 263, 264, 265, 267, 268, 269, 270, 271, 272, 273, 275, 277, 279, 280, 281, 282, 284, 285, 287, 288, 289, 290, 291, 293, 294, 295, 296, 297, 298, 299, 300, 301, 303, 305, 306, 307, 308, 310, 312, 313, 315, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 337, 338, 339, 341, 342, 343, 344, 345, 346, 348, 349, 350, 351, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 372, 373, 374, 375, 376, 378, 379, 382, 384, 385, 387, 388, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 433, 434, 436, 437, 438, 439, 440, 441, 442, 443, 445, 446, 447, 448, 449, 450, 451, 453, 455, 456, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 482, 483, 485, 486, 488, 491, 492, 493, 494, 495, 496, 497, 498, 500, 501, 502, 503, 505, 507, 508, 509, 510, 511, 513, 515, 516, 517, 519]:
            continue
        if item["modality"] == "video" and item["label"] == "harmful" and int(item["id"]) not in [2, 3, 4, 8, 9, 12, 13, 14, 15, 16, 18, 20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 45, 49, 51, 53, 54, 55, 57, 60, 61, 63, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 81, 83, 84, 87, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 115, 116, 119, 121, 123, 126, 127, 128, 129, 131, 135, 138, 139, 141, 142, 143, 145, 146, 148, 151, 153, 154, 155, 156, 157, 158, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 174, 175, 178, 179, 180, 182, 183, 185, 186, 187, 188, 194, 195, 196, 198, 199, 200, 201, 202, 203, 206, 207, 208, 210, 211, 212, 214, 215, 217, 219, 220, 221, 223, 224, 225, 226, 227, 228, 231, 232, 233, 235, 236, 238, 239, 244, 245, 246, 247, 248, 249, 252, 253, 254, 255, 256, 259, 262, 263, 264, 267, 268, 269, 270, 272, 274, 275, 276, 278, 279, 280, 281, 282, 283, 284, 285, 288, 290, 292, 293, 294, 296, 297, 299, 300, 301, 303, 307, 312, 313, 314, 316, 321, 322, 324, 325, 327, 329, 330, 331, 333, 334, 339, 340, 341, 344, 345, 346, 347, 348, 349, 351, 355, 356, 357, 358, 361, 362, 363, 364, 365, 366, 368, 374, 375, 376, 378, 379, 380, 384, 386, 387, 388, 389, 391, 392, 393, 394, 396, 397, 398, 399, 400, 402, 403, 404, 405, 407, 410, 411, 412, 413, 415, 416, 418, 419, 421, 422, 424, 425, 427, 428, 429, 430, 433, 434, 435, 436, 437, 440, 441, 442, 443, 445, 447, 448, 449, 452, 453, 454, 455, 456, 458, 459, 460, 461, 463, 464, 465, 469, 470, 472, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 490, 492, 493, 494, 495, 496, 497, 499, 500, 501, 502, 503, 504, 505, 508, 509, 510, 511, 512, 516, 518, 519]:
            continue
        modality = item.get("modality", "")
        label = item.get("label", "harmful")
        
        if modality not in target_modalities:
            continue
            
        if counts[modality][label] >= max_samples_per_type:
            continue
            
        content = []
        if modality == "image":
            content.append({"type": "image", "image": item["image_path"]})                          
        elif modality == "audio":
            content.append({"type": "audio", "audio": item["audio_path"]})
        elif modality == "video":
            content.append({"type": "video", "video": item["video_path"]})
        
        content.append({"type": "text", "text": item["text_instruction"]})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": content}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, False) # is_train=False
        
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model.thinker(**inputs)
            
        if modality not in mm_activations:
            mm_activations[modality] = {}
            
        for layer_idx in target_layers:
            act = hook_manager.activations[layer_idx][:, -1, :]
            
            if layer_idx not in mm_activations[modality]:
                mm_activations[modality][layer_idx] = {"safe": [], "harmful": []}
                
            mm_activations[modality][layer_idx][label].append(act)
            
        hook_manager.activations = {} # Clear
        counts[modality][label] += 1

    v_mm_dict = {} # v_mm_dict[modality][layer] = vector
    
    for modality in target_modalities:
        if modality not in mm_activations:
            continue
            
        print(f"Calculating vectors for modality: {modality}")
        v_mm_dict[modality] = {}
        
        for layer_idx in target_layers:
            if layer_idx not in mm_activations[modality]:
                continue
                
            safe_acts = mm_activations[modality][layer_idx]["safe"]
            harm_acts = mm_activations[modality][layer_idx]["harmful"]
            
            if len(safe_acts) > 0 and len(harm_acts) > 0:
                v_mm_dict[modality][layer_idx] = calculate_refusal_vector(safe_acts, harm_acts)
            else:
                print(f"  Layer {layer_idx}: Insufficient data (Safe: {len(safe_acts)}, Harmful: {len(harm_acts)})")

    # ==========================================
    # ==========================================
    print("\nStep 3: Comparing V_text and V_mm...")
    similarity_results = {}
    
    for modality in v_mm_dict:
        similarity_results[modality] = {}
        print(f"\n--- Modality: {modality} ---")
        
        for layer_idx in target_layers:
            if layer_idx in v_text and layer_idx in v_mm_dict[modality]:
                vec_t = v_text[layer_idx]
                vec_m = v_mm_dict[modality][layer_idx]
                
                cos_sim = F.cosine_similarity(vec_t.unsqueeze(0), vec_m.unsqueeze(0)).item()
                
                norm_t = torch.norm(vec_t)
                norm_m = torch.norm(vec_m)
                norm_ratio = (norm_m / (norm_t + 1e-6)).item()

                projection = torch.dot(vec_m, vec_t) / (norm_t + 1e-6)
                
                proj_ratio = projection / (norm_t + 1e-6)
                
                similarity_results[modality][layer_idx] = {
                    "cosine_similarity": cos_sim,
                    "norm_ratio": norm_ratio,
                    "projection": projection.item(),
                    "projection_ratio": proj_ratio.item(),
                    "norm_text": norm_t.item(),
                    "norm_mm": norm_m.item()
                }
                
                print(f"Layer {layer_idx}: Cos Sim = {cos_sim:.4f}, Norm Ratio = {norm_ratio:.4f}, Proj Ratio = {proj_ratio.item():.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "vector_similarity_seg.json")
    with open(output_file, 'w') as f:
        json.dump(similarity_results, f, indent=2)
    print(f"向量相似度分析结果已保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Qwen2.5-Omni 模型路径")
    parser.add_argument("--data_file", type=str, required=True, help="实验数据 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="output/similarity", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    
    args = parser.parse_args()
    
    run_vector_similarity_experiment(args)
