#!/bin/bash
#conda activate base
folder1="potus_50t"
folder2="potus_100t"
folder3="potus_120t"

datafolder="potus_data"
rootfolder="/home/azureuser/cloudfiles/code/data/processing/potus/experiment/"
echo $rootfolder"pipeline_log.txt"

log_file=$rootfolder"pipeline_log.txt" 
echo $log_file
echo "start" > $log_file

echo "data folder: "$rootfolder$datafolder"/*.*" 

echo "copy from "$datafolder" to "$folder1 
cp $rootfolder$datafolder $rootfolder$folder1 -r
python potus_pipe_1_gensimlda_01_create_lda.py --experiment $folder1 --num_topics 50 --iterations 200 --passes 8
python potus_pipe_1_gensimlda_02_topics_for_docs.py --experiment $folder1
python potus_pipe_1_gensimlda_03_joined_df.py --experiment $folder1
python potus_pipe_2_postprocess_01_kld.py --experiment $folder1 --Nw 25 --Tw 25 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder1 --Nw 50 --Tw 50 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder1 --Nw 75 --Tw 75 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder1 --Nw 100 --Tw 100
python potus_pipe_2_postprocess_01_kld.py --experiment $folder1 --Nw 125 --Tw 125
echo $folder1" completed "$(date +%Y%m%d%H%M%S) >> $log_file


echo "copy from "$datafolder" to "$folder2
cp $rootfolder$datafolder $rootfolder$folder2 -r
python potus_pipe_1_gensimlda_01_create_lda.py --experiment $folder2 --num_topics 100 --iterations 200 --passes 8
python potus_pipe_1_gensimlda_02_topics_for_docs.py --experiment $folder2
python potus_pipe_1_gensimlda_03_joined_df.py --experiment $folder2
python potus_pipe_2_postprocess_01_kld.py --experiment $folder2 --Nw 25 --Tw 25 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder2 --Nw 50 --Tw 50 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder2 --Nw 75 --Tw 75 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder2 --Nw 100 --Tw 100
python potus_pipe_2_postprocess_01_kld.py --experiment $folder2 --Nw 125 --Tw 125
echo $folder2" completed "$(date +%Y%m%d%H%M%S) >> $log_file


echo "copy from "$datafolder" to "$folder3
cp $rootfolder$datafolder $rootfolder$folder3 -r
python potus_pipe_1_gensimlda_01_create_lda.py --experiment $folder3 --num_topics 120 --iterations 200 --passes 8
python potus_pipe_1_gensimlda_02_topics_for_docs.py --experiment $folder3
python potus_pipe_1_gensimlda_03_joined_df.py --experiment $folder3
python potus_pipe_2_postprocess_01_kld.py --experiment $folder3 --Nw 25 --Tw 25 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder3 --Nw 50 --Tw 50 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder3 --Nw 75 --Tw 75 
python potus_pipe_2_postprocess_01_kld.py --experiment $folder3 --Nw 100 --Tw 100
python potus_pipe_2_postprocess_01_kld.py --experiment $folder3 --Nw 125 --Tw 125
echo $folder3" completed "$(date +%Y%m%d%H%M%S) >> $log_file