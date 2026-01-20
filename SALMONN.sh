#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
for r in {6..7}; do
    conda activate MiniCPM
    CUDA_VISIBLE_DEVICES=6 python Vicuna_test.py --round $r
    conda deactivate 

    cd CosyVoice
    conda activate cosyvoice
    CUDA_VISIBLE_DEVICES=6 python round_generation.py --round $r
    conda deactivate
    cd ..

    conda activate SALMONN
    CUDA_VISIBLE_DEVICES=6 python SALMONN_SPML.py --round $r
    conda deactivate
done
