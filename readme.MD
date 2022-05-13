

compared to medical_net_resnet50 cls_3d is a small model, however it obtained good performance too.
model ensembling(unweighted average) by combining  medical_net_resnet50 and cls_3d two models achieved better performance.

the performance of M0/M1M2 is very good(both multi class and binary class classification).
the performance of M1/M2 is acceptable.

shell scripts:
train.sh
predict.sh
predict_heatmaps_dr.sh
predict_heatmaps_explain.sh

This project includes the following directories:
1.data_process
2.datafiles
3.libs
4.train
5.predict

