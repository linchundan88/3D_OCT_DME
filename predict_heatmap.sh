#!/bin/bash

data_version="v1"
dir_dest_base="/tmp2/3D_OCT_DME/2022_4_28/"

for task_type in "3D_OCT_DME_M0_M1M2_test"  #"3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M1_M2_test"
do
  for heatmap_type in "GuidedBackprop"  # "IntegratedGradients" "GuidedBackprop"  "GuidedGradCam"
  do
    for model_name in "cls_3d" "medical_net_resnet50"
    do
      echo "generating heatmaps:${heatmap_type} ${task_type} ${data_version} model_name:${model_name} \
        dir_dest:${dir_dest}/${task_type}/${model_name}/${heatmap_type}"
      python ./predict/binary_class/my_gen_heatmap.py --heatmap_type ${heatmap_type} \
        --task_type ${task_type} --data_version ${data_version} --model_name ${model_name} \
        --dir_dest ${dir_dest_base}/${task_type}/${model_name}/${heatmap_type}
    done
  done
done



echo "predict heatmaps completed."
