#!/bin/bash
#conda activate base
datafolder="potus_data"


python potus_pipe_0_preprocess_01.py --experiment $datafolder
python potus_pipe_0_preprocess_02_tokenise.py --experiment $datafolder
python potus_pipe_0_preprocess_03_bigrams.py --experiment $datafolder

echo "data completed "$(date +%Y%m%d%H%M%S) >> ~/cloudfiles/code/data/processing/potus/experiment/eval_progress.txt
