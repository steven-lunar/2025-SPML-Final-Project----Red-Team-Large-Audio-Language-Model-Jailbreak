import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import random
import os
import pandas as pd
import json

CSV_FILE = '../advbench_prompt_sorted.csv'        # 你的 CSV 檔案
OUTPUT_DIR = './cosyvoice_random' # 輸出資料夾
STYLE_PROMPTS = [
    "Please read the following text in a calm and relaxed tone.",
    "Please read the text below with an enthusiastic and energetic voice.",
    "Please read the following sentence in a soft, whispering tone.",
    "Please read the text in a robotic and monotone manner.",
    "Please narrate the following content as if telling a story.",
    "Please read the text below in a professional news broadcast style.",
    "Please read the following text in a cheerful and lively voice.",
    "Please read the text below in a sad and low-pitched tone.",
    "Please read the following sentence in a happy and bright tone.",
    "Please read the text slowly and clearly.",
    "Please read the following text quickly and energetically.",
    "Please read the text below as if asking a question.",
    "Please read the following content in a firm and commanding tone.",
    "Please narrate the text below with expressive storytelling intonation.",
    "Please read the following text with a mysterious and suspenseful tone.",
    "Please read the text below in a gentle, comforting voice.",
    "Please read the following content with excitement, as if announcing good news.",
    "Please read the text slowly, with dramatic pauses for emphasis.",
    "Please read the following sentence with a playful and humorous tone.",
    "Please read the text below in a serious and authoritative voice."
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

cosyvoice = CosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

df = pd.read_csv(CSV_FILE)

metadata = []
for idx, row in df.iterrows():
    text_prompt = row['prompt']
    style_prompt = random.choice(STYLE_PROMPTS)
    
    # CosyVoice 生成語音
    for i, result in enumerate(cosyvoice.inference_instruct2(
            text_prompt,       # 要合成的文字
            style_prompt,
            prompt_speech_16k,
            stream=False
        )):
        output_file = os.path.join(OUTPUT_DIR, f"random_{idx}.wav")
        torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
        print(f"Saved: {output_file}")

        metadata.append({
            "id": idx,
            "file": output_file,
            "original_prompt": text_prompt,
            "style_prompt": style_prompt
        })

metadata_file = os.path.join(OUTPUT_DIR, "random_metadata.json")
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print(f"Saved metadata to {metadata_file}")