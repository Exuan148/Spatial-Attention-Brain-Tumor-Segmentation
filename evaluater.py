#START: OWN CODE
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from inputs import create_input
import pynvml
from u_net3d import U_Net
import os
import datetime
from config.u_net3d_config import *
import numpy as np
from preprocessing import mask_to_onehot
# todo fetch GPU
pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpulist = []
for i in range(deviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)  
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if (meminfo.used / (1024 ** 2) <= 320):
        print("GPU:{i} is available".format(i=i))
        gpulist.append(i)
        if(len(gpulist)==1):
            break
if len(gpulist) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for i in range(len(gpulist) - 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] + str(gpulist[i]) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] + str(gpulist[-1])
print(os.environ["CUDA_VISIBLE_DEVICES"])


from tqdm import tqdm
import nibabel as nib
IMG_DATA_PATH="BRATSimg/imagesTest"
IMG_LABEL_PATH="BRATSimg/labelsTest"
def evaluater():
    DISTRIIBUTED_STRATEGY_GPUS = ["gpu:{i}".format(i=i) for i in range(1)]
    mirrored_strategy = tf.distribute.MirroredStrategy(DISTRIIBUTED_STRATEGY_GPUS)
    with mirrored_strategy.scope():
        # todo initialize model
        model = U_Net()
        model.unet.load_weights(os.path.join(CHECKPOINTS_PATH, 'BrainTumor_0.12_0.34_1980.h5'))

        # todo find files
        sample_names = []
        for i, j, k in os.walk(IMG_DATA_PATH):
            for file in k:
                if file.startswith("BRATS") and file.endswith('.gz'):
                    sample_names.append(file)

        for sample_name in tqdm(sample_names, desc='evaluate: '):
            # todo load image
            nii_img = nib.load(os.path.join(IMG_DATA_PATH, sample_name))
            nii_fdata_img = nii_img.get_fdata()
            image = np.pad(nii_fdata_img, ((0, 0), (0, 0), (5, 0), (0, 0)),
                           mode='constant', constant_values=0)[40:200, 14:206, :, :]
            image = image.astype(np.float32)
            image = tf.image.per_image_standardization(image)  
            image=image.numpy()
            image=np.reshape(image,(1,160,192,160,4))
            image = image.astype(np.float32)
            outputs = model.unet.predict(image)
            np.save(os.path.join(OUTPUTS_PATH,sample_name.replace('.nii.gz','')),outputs)


def metrics():
    # todo find files
    pre_label_names = []
    for i, j, k in os.walk(OUTPUTS_PATH):
        for file in k:
            pre_label_names.append(file.replace('.npy',''))

    dice_coefs_list=[]

    for pre_label_name in tqdm(pre_label_names,desc='metrics: '):
        # todo load predicted labels
        pre_label=np.load(os.path.join(OUTPUTS_PATH,pre_label_name+".npy"))

        # todo flatting and arg-max
        pre_label=np.reshape(pre_label,(160*192*160,4))
        idx = np.argmax(pre_label, axis=1)
        pre_label[idx == 0] = np.array([1, 0, 0, 0])
        pre_label[idx == 1] = np.array([0, 1, 0, 0])
        pre_label[idx == 2] = np.array([0, 0, 1, 0])
        pre_label[idx == 3] = np.array([0, 0, 0, 1])
        pre_label.astype(np.float32)

        # todo load true labels
        nii_label = nib.load(os.path.join(IMG_LABEL_PATH, pre_label_name+".nii.gz"))
        nii_fdata_label = nii_label.get_fdata()
        tru_label = np.pad(nii_fdata_label, ((0, 0), (0, 0), (5, 0)),
                       mode='constant', constant_values=0)[40:200, 14:206, :]
        tru_label = np.reshape(tru_label, (160, 192, 160, 1))
        tru_label=tru_label.astype(np.float32)
        palette = [[0], [1], [2], [3]]
        tru_label = mask_to_onehot(tru_label, palette)
        tru_label = np.reshape(tru_label, (160*192*160,4))

        # todo calculate dice loss
        pre_label=tf.convert_to_tensor(pre_label)
        tru_label=tf.convert_to_tensor(tru_label)
        intersection=tru_label*pre_label
        intersection = tf.reduce_sum(intersection, axis=0)
        union = tru_label + pre_label
        union = tf.reduce_sum(union, axis=0)
        smooth = 1.e-5
        dice_coefs=2*(intersection+smooth)/(union+smooth)
        dice_coefs=dice_coefs.numpy()

        # todo save result
        np.save(os.path.join(RESULTS_PATH,pre_label_name+".npy"),dice_coefs)
        dice_coefs_list.append(dice_coefs.copy())

    dice_coefs=np.array(dice_coefs)
    print(np.mean(dice_coefs_list,axis=0))







if __name__ == '__main__':
    evaluater()
    metrics()

#END: OWN CODE