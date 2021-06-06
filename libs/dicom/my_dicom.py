
import os
import pydicom
import numpy as np
import cv2


# input dicom file, output 3d ndarray
def get_dicom_array(filename):
    # OCT: (128, 885, 512), (D,H,W) Fundus image:(2000, 2992, 3)
    ds2 = pydicom.dcmread(filename)
    array1 = ds2.pixel_array
    return array1


def crop_3d_topocon(array1,
                    image_shape=(128, 128), padding_square=True,
                    depth_interval=1, remainder=0):

    # input (128,885,512)
    if depth_interval > 1:  # only keep some depth channels
        list_images = []
        for i in range(array1.shape[0]):
            if i % depth_interval == remainder:
                tmp_array = array1[i, :, :]
                list_images.append(tmp_array)

        assert len(list_images) > 0, 'dicom depth error!'
        array_subsampoling_z = np.stack(list_images, axis=0)
    else:
        array_subsampoling_z = array1

    # region Vertical cutting
    array_tmp = np.sum(array_subsampoling_z, axis=(0, 2))  # (D,H,W)
    max_h = np.argmax(array_tmp)

    height, width = array1.shape[1:3]  # (D,H,W)
    bottom = max(0, max_h-width//2)
    top = min(max_h + width//2, height)

    if max_h-width//2 < 0:
        top += abs(max_h-width//2)
    if max_h + width//2 > height:
        bottom -= (max_h + width//2 - height)

    array_cropping = array_subsampoling_z[:, bottom:top, :]

    #endregion

    #zero padding if necessary, in order to get square
    if padding_square:
        list_images = []
        from libs.img_preprocess.my_image_helper import image_to_square
        for i in range(array_cropping.shape[0]):
            tmp_image = array_cropping[i, :, :]
            list_images.append(image_to_square(tmp_image))

        array_cropping = np.stack(list_images, axis=0)

    if image_shape is not None:
        list_images = []
        for i in range(array_cropping.shape[0]):
            tmp_image = cv2.resize(array_cropping[i, :, :],
                                   (image_shape[1], image_shape[0]))
            list_images.append(tmp_image)

        array_cropping = np.stack(list_images, axis=0)

    return array_cropping


# Topocon OCT file size bigger than THRETHOLD_FILESIZE
THRETHOLD_FILESIZE = 5*1024*1024

def is_oct_file(img_file_source):
    _, file_extension = os.path.splitext(img_file_source)
    if file_extension.upper() not in ['.DCM', '.DICOM']:
        return False
    if os.path.getsize(img_file_source) < THRETHOLD_FILESIZE:  # 5MB
        return False

    try:
        array1 = get_dicom_array(img_file_source)
    except:
        return False

    if array1.shape[2] == 3:  # color fundus image
        return False
    else:
        return True


def dicom_save_dirs(source_dir, dir_dest_base,
                    save_npy=True, save_image_files=True, image_shape=None,
                    padding_square=True,
                    depth_interval=1, remainder=0):

    for dir_path, _, files in os.walk(source_dir, False):
        for f in files:
            file_dicom = os.path.join(dir_path, f)
            if not is_oct_file(file_dicom):
                print('not a OCT dicom file:', file_dicom)
                continue
            else:
                print('processing OCT file:{}'.format(file_dicom))

            array1 = get_dicom_array(file_dicom) #topocon (128,885,512) (D,H,W)

            _, filename = os.path.split(file_dicom)
            filename_stem = os.path.splitext(filename)[0]
            dir_dest = os.path.join(dir_dest_base, filename_stem)

            pat_id = f.replace('.dcm', '')
            pat_id = pat_id.replace('.DCM', '')
            pat_id = pat_id.replace('.dicom', '')
            pat_id = pat_id.replace('.dicom', '')

            if image_shape is not None:
                array1 = crop_3d_topocon(array1,
                                         image_shape=image_shape, padding_square=padding_square,
                                         depth_interval=depth_interval, remainder=remainder)

            if save_npy:
                if depth_interval == 1 and remainder == 0:
                    filename = os.path.join(dir_dest, pat_id + '.npy')
                else:
                    filename = os.path.join(dir_dest, pat_id + f'_d{depth_interval}_r{str(remainder)}.npy')

                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, array1)

            if save_image_files:
                array1 = crop_3d_topocon(array1,
                                         image_shape=image_shape, padding_square=padding_square)
                for i in range(array1.shape[0]):  # (D,H,W)
                    filename = os.path.join(dir_dest, f'{str(i)}.png')
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    cv2.imwrite(filename, array1[i, :, :])


def slices_to_npy(dir1, depth_ratio=1, remainder=0, slice_num=128):

    for root, dirs, _ in os.walk(dir1, False):
        #zesis 1.jpeg, topocon 0.png
        for dir_tmp in dirs:
            dir_path = os.path.join(root, dir_tmp)
            if os.path.exists(os.path.join(dir_path, '1.jpeg')) \
                    or os.path.exists(os.path.join(dir_path, '0.png')):

                print(dir_path)

                list_images = []
                for i in range(slice_num):
                    if i % depth_ratio == remainder:
                        if os.path.exists(os.path.join(dir_path, f'{str(i)}.png')):
                            img_file = os.path.join(dir_path,  f'{str(i)}.png')
                        elif os.path.exists(os.path.join(dir_path, f'{str(i+1)}.jpeg')):
                            img_file = os.path.join(dir_path, f'{str(i+1)}.jpeg')
                        else:
                            # raise Exception(f'file not found:{img_file}')
                            break

                        img1 = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) #(H,W)
                        list_images.append(img1)

                if len(list_images) != slice_num:
                    print(f'error:{dir_path}')

                images = np.stack(list_images, axis=0)  #(H,W)->(D,H,W)

                pat_id = dir_path.split('/')[-1]
                if depth_ratio == 1 and remainder == 0:
                    file_npy = os.path.join(dir_path, pat_id + '.npy')
                else:
                    file_npy = os.path.join(dir_path, pat_id + f'_d{depth_ratio}_r{str(remainder)}.npy')
                # print(file_npy)
                np.save(file_npy, images)
