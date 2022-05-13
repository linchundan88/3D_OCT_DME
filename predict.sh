#! /bin/bash


DIR_DEST="/tmp2/3D_OCT_DME/2022_4_30"

for TASK_TYPE in "3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M0_M1M2_test"
do
  DATA_VERSION="v1"
  echo "predicting $TASK_TYPE save to ${DIR_DEST}/${TASK_TYPE}"
  python ./predict/binary_class/my_predict.py --task_type ${TASK_TYPE} --data_version ${DATA_VERSION} \
    --image_shape 128 128  --dir_dest ${DIR_DEST}/${TASK_TYPE}  --export_confusion_files
done


for TASK_TYPE in "3D_OCT_DME_M1_M2_train" "3D_OCT_DME_M1_M2_valid" "3D_OCT_DME_M1_M2_test"
do
  DATA_VERSION="v1"
  echo "predicting $TASK_TYPE save to ${DIR_DEST}/${TASK_TYPE}"
  python ./predict/binary_class/my_predict.py --task_type ${TASK_TYPE} --data_version ${DATA_VERSION} \
    --image_shape 128 128  --dir_dest ${DIR_DEST}/${TASK_TYPE}  --export_confusion_files
done


echo "prediction completed."
