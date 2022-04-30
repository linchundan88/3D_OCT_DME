import os
import shutil

#npy converted to image dir
def export_confusion_files_binary_class(image_files, labels_gt, probs, dir_original, dir_preprocess, dir_dest, threshold=0.5):
    for image_file, label_gt, prob in zip(image_files, labels_gt, probs):
        if prob > threshold:
            label_pred = 1
        else:
            label_pred = 0
        if label_gt != label_pred:
            dir_base = os.path.dirname(image_file.replace(dir_preprocess, dir_original))
            for dir_path, _, files in os.walk(dir_base, False):
                for f in files:
                    file_full_path = os.path.join(dir_path, f)
                    _, file_name = os.path.split(file_full_path)
                    _, file_ext = os.path.splitext(file_name)
                    if file_ext.lower() in ['.jpeg', '.jpg', '.png']:
                        file_partial_path = file_full_path.replace(dir_original, '')
                        file_name1 = os.path.join(dir_dest, f'{label_gt}_{label_pred}', file_partial_path)
                        tmp_dir, tmp_filename = os.path.split(file_name1)
                        if label_gt == 0:
                            dir_prob = f'prob{int((1 - prob) * 100)}_'
                        if label_gt == 1:
                            dir_prob = f'prob{int(prob * 100)}_'

                        list_tmp_dir = tmp_dir.split('/')
                        list_tmp_dir[-1] = dir_prob + list_tmp_dir[-1]
                        img_dest = os.path.join('/'.join(list_tmp_dir), tmp_filename)
                        print(img_dest)
                        os.makedirs(os.path.dirname(img_dest), exist_ok=True)
                        shutil.copy(file_full_path, img_dest)