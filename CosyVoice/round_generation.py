import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import random
import os
import pandas as pd
import json
import argparse

import torch

# 要佔用的 GB 數
GB = 30
bytes_to_alloc = GB * 1024**3

# float32 是 4 bytes，所以算出要幾個元素
num_elems = bytes_to_alloc // 4

# 直接 allocate
dummy = torch.empty(num_elems, dtype=torch.float32, device="cuda")

cosyvoice = CosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)


output_metadata = []
def cosyvoice_generate(round_num):
	OUTPUT_DIR = f'../SPML_output/round{round_num}/audio/SALMONN-7B-T'
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	CSV_FILE = '../advbench_prompt_sorted.csv'  
	df = pd.read_csv(CSV_FILE)
	base = []
	with open("../cosyvoice_random/random_metadata.json", "r") as f:
		base = json.load(f)

	metadata = []
	with open(f"../SPML_output/round{round_num}/SALMONN-7B_vicuna_refinement.json") as f:
		metadata = json.load(f)["refinements"]

	for idx, row in df.iterrows():
		text_prompt = row['prompt']
		# style_prompt = metadata[idx]["style prompt"]
		style_prompt = base[idx]["style_prompt"]

		for i, result in enumerate(cosyvoice.inference_instruct2(
			metadata[idx]["text prompt"],       # 要合成的文字
			style_prompt,
			prompt_speech_16k,
			stream=False
		)):
			output_file = os.path.join(OUTPUT_DIR, f"round{round_num}_{idx}.wav")
			torchaudio.save(output_file, result['tts_speech'], cosyvoice.sample_rate)
			print(f"Saved: {output_file}")

			output_metadata.append({
				"id": idx,
				"file": output_file,
				"original_prompt": text_prompt,
				"style_prompt": style_prompt,
				"refined_prompt": metadata[idx]["text prompt"]
			})
		# break

	output_metadata_file = os.path.join(OUTPUT_DIR, f"SALMONN-7B-T_round{round_num}_metadata.json")
	with open(output_metadata_file, "w", encoding="utf-8") as f:
		json.dump(output_metadata, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--round", type=int, required=True)
	args = parser.parse_args()
	cosyvoice_generate(args.round)