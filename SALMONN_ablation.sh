#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
for r in {1..7}; do
    cd CosyVoice
    conda activate cosyvoice
    CUDA_VISIBLE_DEVICES=6 python round_generation.py --round $r
    conda deactivate
    cd ..

    conda activate SALMONN
    CUDA_VISIBLE_DEVICES=6 python SALMONN_SPML.py --round $r
    conda deactivate
done