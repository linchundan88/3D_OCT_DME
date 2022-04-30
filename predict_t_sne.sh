#!/bin/bash

data_version="v1"
dir_dest_base="/tmp2/3D_OCT_DME/2022_4_28/t-sne"

for task_type in "3D_OCT_DME_M0_M1M2_test"  #"3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M0_M1M2_test"
do
  for model_name in "medical_net_resnet50"  #"cls_3d" "medical_net_resnet50"
  do
    echo "generating t_SNE:${task_type} data_version:${data_version} \
      model_name:${model_name} dir_dest:${dir_dest_base}/${task_type}/${model_name}"
    python ./predict/binary_class/my_gen_t_SNE.py --task_type ${task_type} --data_version ${data_version} \
      --model_name ${model_name} --dir_dest ${dir_dest_base}/${task_type}/${model_name}
  done

done



echo "predict t-SNE heatmaps completed."
