#! /bin/bash
#generating dimension reduction heat-maps.

DIR_DEST="/tmp2/3D_OCT_DME/2022_5_7"
DATA_VERSION="v1"
DR_METHOD='tsne'  #tsne umap
LABELS_TEXT="non-CI-DME CI-DME"  #Normal DME  vs non-CI-DME CI-DME

#"3D_OCT_DME_M0_M1M2_train" "3D_OCT_DME_M0_M1M2_valid" "3D_OCT_DME_M0_M1M2_test"
#"3D_OCT_DME_M1_M2_train" "3D_OCT_DME_M1_M2_valid" "3D_OCT_DME_M1_M2_test"
for TASK_TYPE in "3D_OCT_DME_M1_M2_test"
do
  for MODEL_NAME in "medical_net_resnet50" "cls_3d" #"medical_net_resnet50"
  do
    echo "generating t_SNE:${TASK_TYPE} data_version:${DATA_VERSION} dr_method:${DR_METHOD} \
      MODEL_NAME:${MODEL_NAME} dir_dest:${DIR_DEST}/${DR_METHOD}/${TASK_TYPE}/${MODEL_NAME}"
    python ./predict/binary_class/my_heatmaps_dr.py --task_type ${TASK_TYPE} --data_version ${DATA_VERSION} \
      --dr_method ${DR_METHOD} --model_name ${MODEL_NAME}  --image_shape 128 128 \
      --dir_dest ${DIR_DEST}/${DR_METHOD}/${TASK_TYPE}/${MODEL_NAME} --labels_text ${LABELS_TEXT}
  done

done


echo "predict t-SNE heatmaps completed."