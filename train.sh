#!/bin/bash

save_model_dir="/tmp2/2022_4_29/binary_classifier_m0_m1m2"
TRAIN_TIMES=5

#for ((i=0; i<=TRAIN_TIMES; i++))
#do
#  predict_type="binary_classifier_m0_m1m2"
#  echo "training ${predict_type} time:${i}"
#
#  for model_name in "medical_net_resnet50"  #"cls_3d" "medical_net_resnet50"
#  do
#    echo "training model ${model_name}"
#    python ./train/my_train_binary_class_m0_m1m2.py --model_name ${model_name} \
#      --save_model_dir ${save_model_dir}/${predict_type}/${model_name}/times${i} --amp
#  done
#done


for ((i=0; i<=TRAIN_TIMES; i++))
do
  predict_type="binary_classifier_m1_m2"
  echo "training ${predict_type} time:${i}"

  for model_name in "cls_3d" "medical_net_resnet50"
  do
    echo "training model ${model_name}"
    python ./train/my_train_binary_class_m0_m1m2.py --model_name ${model_name} \
      --save_model_dir ${save_model_dir}/${predict_type}/${model_name}/times${i} --amp
  done
done