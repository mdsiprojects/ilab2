#!/bin/bash
conda activate base
folder="eval_passes_at_50"
speeches_list="/home/azureuser/cloudfiles/code/Users/Shared/hansard/corpus/hansard_speech_records_parse_20_10_03-01_19_24.pkl"

python hansard_df_preprocess_01_df_create.py --experiment $folder --filename $speeches_list
python hansard_df_preprocess_02_tokenise.py --experiment $folder
python hansard_df_preprocess_03_bigrams.py --experiment $folder

python hansard_topics_gensimlda_01_create_lda.py --experiment $folder --num_topics 100 --iterations 50 --passes 8
python hansard_topics_gensimlda_02_topics_for_docs.py --experiment $folder
python hansard_topics_gensimlda_03_joined_df.py --experiment $folder

python hansard_postprocess_kld.py --experiment $folder --Nw 15 --Tw 15
python hansard_postprocess_kld.py --experiment $folder --Nw 1500 --Tw 1500
python hansard_postprocess_kld.py --experiment $folder --Nw 500 --Tw 500



