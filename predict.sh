#!/bin/bash


dir_dest="/tmp2/3D_OCT_DME/2022_4_30"

for task_type in "3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M0_M1M2_test"
do
  data_version="v1"
  echo $task_type
  echo "predicting $task_type save to ${dir_dest}/${task_type}"
  python ./predict/binary_class/my_predict.py --task_type ${task_type} --data_version ${data_version} \
    --dir_dest ${dir_dest}/${task_type}  #--export_confusion_files
done


for task_type in "3D_OCT_DME_M1_M2_train" "3D_OCT_DME_M1_M2_valid" "3D_OCT_DME_M1_M2_test"
do
  data_version="v1"
  echo $task_type
  echo "predicting $task_type save to ${dir_dest}/${task_type}"
  python ./predict/binary_class/my_predict.py --task_type ${task_type} --data_version ${data_version} \
    --dir_dest ${dir_dest}/${task_type}  #--export_confusion_files
done



echo "prediction completed."
