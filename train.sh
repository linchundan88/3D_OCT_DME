#!/bin/bash

save_model_dir="/tmp2/2022_4_30"
TRAIN_TIMES=20

#for ((i=0; i<=TRAIN_TIMES; i++))
#do
#  train_type="binary_classifier_m0_m1m2"
#  for model_name in "medical_net_resnet50"  #"cls_3d" "medical_net_resnet50"
#  do
#    echo "training ${train_type} time:${i}" "model:${model_name}"
#    python ./train/my_train_binary_class_m0_m1m2.py --model_name ${model_name} \
#      --save_model_dir ${save_model_dir}/${train_type}/${model_name}/times${i} --amp
#  done
#done


for ((i=0; i<=TRAIN_TIMES; i++))
do
  train_type="binary_classifier_m1_m2"
  for model_name in "cls_3d" "medical_net_resnet50"
  do
    echo "training ${train_type} time:${i}" "model:${model_name}"
    python ./train/my_train_binary_class_m1_m2.py --model_name ${model_name} \
      --save_model_dir ${save_model_dir}/${train_type}/${model_name}/times${i} \
      --image_shape 128 128 --amp --sampling_weights 1 2 --pos_weight 2
  done
done