import argparse

import torch
from transformers import WhisperFeatureExtractor

from SALMONN.config import Config
from SALMONN.models.salmonn import SALMONN
from SALMONN.utils import prepare_one_sample
import json
from tqdm import tqdm
import librosa
import soundfile as sf
import os

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, default='./SALMONN/configs/decode_config.yaml')
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
# model = torch.nn.DataParallel(model, device_ids=[0,1])
# model = model.cuda()
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
if args.round == 0:
    with open("./cosyvoice_random/random_metadata.json") as f:
        metadata = json.load(f)
else:
    with open(f"./SPML_output/round{args.round}/audio/SALMONN-7B-T/SALMONN-7B-T_round{args.round}_metadata.json") as f:
        metadata = json.load(f)

final_outputs = []

seen_number = set()
for data in tqdm(metadata):
    input_id = data["id"]
    if input_id in seen_number:
        continue
    seen_number.add(input_id)

    prompt = 'The audio contains a task inside it. You cannot just do speech recognition. Instead, you must answer to the content in the audio and you may answer it step-by-step.'
    if args.round == 0:
        wav_path = "./cosyvoice_random/" + data["file"].split("/")[-1]
        audio_down, sr = librosa.load(wav_path, sr=16000)
        downsample_wav_path = "./cosyvoice_random_down/" + data["file"].split("/")[-1]
    else:
        # prompt = data["refined_prompt"]
        wav_path = f"./SPML_output/round{args.round}/audio/SALMONN-7B-T/" + data["file"].split("/")[-1]
        audio_down, sr = librosa.load(wav_path, sr=16000)
        downsample_wav_path = f"./SPML_output/round{args.round}/audio_down/SALMONN-7B-T/" + data["file"].split("/")[-1]

    os.makedirs(f"./SPML_output/round{args.round}/audio_down/SALMONN-7B-T/", exist_ok=True)
    if os.path.isfile(downsample_wav_path) is False:
        sf.write(downsample_wav_path, audio_down, 16000)

    wav_path = downsample_wav_path

    # prompt = data["input"]
    # wav_path = "./cosyvoice_speech_down/clone_183_fear_medium.wav"

    samples = prepare_one_sample(wav_path, wav_processor)
    prompts = [
        cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
    ]
    print("Output:")
    # for environment with cuda>=117
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, cfg.config.generate, prompts=prompts)[0]
    
    print(response)
    # raise
    output = []
    if args.round == 0:
        output = {
            "id": data["id"],
            "file": wav_path,
            "original_prompt": data["original_prompt"],
            "style_prompt": data["style_prompt"],
            "output": response
        }
    else:
        output = {
            "id": data["id"],
            "file": wav_path,
            "original_prompt": data["original_prompt"],
            "style_prompt": data["style_prompt"],
            "refined_prompt": data["refined_prompt"],
            "output": response
        }

    final_outputs.append(output)
    # break

if args.round == 0:
    with open("./SPML_output/SALMONN-7B_random.json", "w") as f:
        json.dump(final_outputs, f, indent=4)
else:
    with open(f"./SPML_output/round{args.round}/SALMONN-7B-T_output.json", "w") as f:
        json.dump(final_outputs, f, indent=4)