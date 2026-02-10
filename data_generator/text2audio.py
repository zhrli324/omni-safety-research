import os
import json
import argparse
import requests
import dashscope
import time
from dashscope.audio.qwen_tts import SpeechSynthesizer

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
API_KEY = ""

def process_tts(input_file, output_dir, output_json):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # data = data[:3]
    
    results = []
    
    for i, item in enumerate(data):
        try:
            # Extract info
            split_result = item.get("split_result", {})
            text_instruction = split_result.get("text_instruction", "")
            audio_content = split_result.get("audio_content", "")
            # audio_content = item.get("text", {})
            
            if not audio_content:
                print(f"Skipping item {i}: No audio content found.")
                continue
                
            print(f"Processing item {i}...")
            
            # Call TTS API
            if i//10%2 == 0:
                response = SpeechSynthesizer.call(
                    model="qwen3-tts-flash",
                    api_key=API_KEY,
                    text=audio_content, 
                    voice="Ryan",
                )
            else:
                response = SpeechSynthesizer.call(
                    model="qwen3-tts-flash-2025-09-18",
                    api_key=API_KEY,
                    text=audio_content, 
                    voice="Ryan",
                    # language_type="Chinese" # Removed to allow auto-detection or English support since input is English
                )
            
            if response.status_code == 200:
                audio_url = response["output"]["audio"]["url"]
                if audio_url:
                    audio_filename = f"{i}.wav"
                    audio_path = os.path.join(output_dir, audio_filename)
                    
                    # Download the audio file
                    audio_res = requests.get(audio_url)
                    if audio_res.status_code == 200:
                        with open(audio_path, 'wb') as f:
                            f.write(audio_res.content)
                        
                        results.append({
                            "text": audio_content,
                            "audio_path": audio_filename,
                            "image_path": f"{i}.png"
                        })
                    else:
                        print(f"Failed to download audio for item {i}")
                else:
                    print(f"No audio URL in response for item {i}")
            else:
                print(f"Failed to generate audio for item {i}: {response.message}")
            time.sleep(2)  # To avoid hitting rate limits
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            
    # Save output JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Results saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch TTS processing using DashScope.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated audio files.")
    parser.add_argument("--output_json", type=str, help="Path to output JSON file.")
    
    args = parser.parse_args()
    
    process_tts(args.input_file, args.output_dir, args.output_json)
