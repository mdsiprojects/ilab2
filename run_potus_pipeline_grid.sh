#!/bin/bash
# #ok so i killed it while it was doing potus_40t_800_10
# 6:12
# so there was just three more permutations left for the 40t
# 6:13
# what i will do is to remove 30 and 40 topics from the search since they are already done .. we can come back to the last remainng permutations of 40t later if we need
echo gridsearch

topics=(120 110 100 90 80 70 60 50)
iterations=(800 500 300 100 700 600 400 200)
passes=(12 10 8 6 4)

datafolder="potus_data"
rootfolder="/home/azureuser/cloudfiles/code/data/processing/potus/experiment/"

log_file=$rootfolder"gridsearch_pipeline_log.txt"
echo $log_file
echo $(date +%Y-%m-%d_%H:%M)"start over !!! " >> $log_file

echo "data folder: "$rootfolder$datafolder"/*.*"

for p in "${passes[@]}"
do
    for it in "${iterations[@]}"
    do
        for t in "${topics[@]}"
        do

            folder="potus_"$t"t_"$it"_"$p
            echo $(date +%Y-%m-%d_%H:%M) $folder" start" >> $log_file
            echo "copy from "$datafolder" to "$folder
            cp $rootfolder$datafolder $rootfolder$folder -r
            python potus_pipe_1_gensimlda_01_create_lda.py --experiment $folder --num_topics $t --iterations $it --passes $p
            python potus_pipe_1_gensimlda_02_topics_for_docs.py --experiment $folder
            python potus_pipe_1_gensimlda_03_joined_df.py --experiment $folder
            python potus_pipe_2_postprocess_01_kld.py --experiment $folder --Nw 25 --Tw 25
            python potus_pipe_2_postprocess_01_kld.py --experiment $folder --Nw 50 --Tw 50
            python potus_pipe_2_postprocess_01_kld.py --experiment $folder --Nw 75 --Tw 75
            python potus_pipe_2_postprocess_01_kld.py --experiment $folder --Nw 100 --Tw 100
            python potus_pipe_2_postprocess_01_kld.py --experiment $folder --Nw 125 --Tw 125
            echo $(date +%Y-%m-%d_%H:%M) $folder" completed" >> $log_file
        done
    done
done