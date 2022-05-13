#! /bin/bash

DIR_DEST="/tmp2/3D_OCT_DME/2022_4_28/"
DATA_VERSION="v1"

for TASK_TYPE in "3D_OCT_DME_M0_M1M2_test"  #"3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M1_M2_test"
do
  for HEATMAP_TYPE in "GuidedBackprop"  # "IntegratedGradients" "GuidedBackprop"  "GuidedGradCam"
  do
    for MODEL_NAME in "cls_3d" "medical_net_resnet50"
    do
      echo "generating heatmaps:${HEATMAP_TYPE} ${TASK_TYPE} ${DATA_VERSION} model_name:${MODEL_NAME} \
        dir_dest:${dir_dest}/${TASK_TYPE}/${MODEL_NAME}/${HEATMAP_TYPE}"
      python ./predict/binary_class/my_heatmap_explain.py --heatmap_type ${HEATMAP_TYPE} \
        --task_type ${TASK_TYPE} --data_version ${DATA_VERSION} --model_name ${MODEL_NAME} \
        --dir_dest ${DIR_DEST}/${TASK_TYPE}/${MODEL_NAME}/${HEATMAP_TYPE}  --image_shape 128 128
    done
  done
done


echo "predict heatmaps completed."
