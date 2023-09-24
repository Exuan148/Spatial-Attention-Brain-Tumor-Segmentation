#START: OWN CODE
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import shutil
import nibabel as nib

np.random.seed(0)
IMG_DATA_PATH="BRATSimg/imagesTrain"
IMG_LABEL_PATH="BRATSimg/labelsTrain"
TF_DATA_PATH="data/"

# todo split dataset as 4:1 for train-set and test-set
 def dataset_split(img_path):
     filenames=[]
     for i, j, k in os.walk(os.path.join(img_path,'imagesTrain/')):
         for file in k:
             if file.startswith("BRATS") and file.endswith('.gz'):
                 filenames.append(file)
     filenames=np.array(filenames)
     np.random.shuffle(filenames)
     for file in filenames[:96]:
         try:
             shutil.move(dst=img_path+'imagesTest/'+file, src=img_path+'imagesTrain/'+file)
             shutil.move(dst=img_path+'labelsTest/'+file, src=img_path+'labelsTrain/'+file)
             print(file," moved")
         except Exception as e:
             print(e,file)

#END: OWN CODE

def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1, and K is the number of segmented class.
    eg:
    mask:label
    palette:[[0],[1],[2],[3],[4],[5]]
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

#START: OWN CODE

def preprocessing(data_path,sample_names):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # todo create 20 different TFRecord files, then fill data randomly
    n_shards=20
    writers=[]
    for i in range(n_shards):
        writers.append(tf.io.TFRecordWriter(
            "{}-{:0>5d}-of-{:0>5d}".format(data_path, i, n_shards)
        ))

    print(data_path)
    for sample_name in tqdm(sample_names,desc='generating: '):
        # todo: load img, img size (216,216,155,4)-->(160, 192, 160, 4)
        nii_img = nib.load(os.path.join(IMG_DATA_PATH,sample_name))
        nii_fdata_img = nii_img.get_fdata()
        image = np.pad(nii_fdata_img, ((0, 0), (0, 0), (5, 0), (0, 0)),
                               mode='constant', constant_values=0)[40:200, 14:206, :, :]
        # todo: load label, label size (216,216,155)-->(160, 192, 160)
        nii_label = nib.load(os.path.join(IMG_LABEL_PATH, sample_name))
        nii_fdata_label = nii_label.get_fdata()
        label = np.pad(nii_fdata_label, ((0, 0), (0, 0), (5, 0)),
                       mode='constant', constant_values=0)[40:200, 14:206, :]
        label=np.reshape(label,(160, 192, 160,1))
        palette=[[0], [1], [2], [3]]
        label=mask_to_onehot(label,palette)

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        image_bytes = image.tobytes()
        label_bytes = label.tobytes()
        features = {}
        features['image_raw'] = _bytes_feature(image_bytes)
        features['label'] = _bytes_feature(label_bytes)
        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()
        writers[np.random.choice([idx for idx in range(20)])].write(tf_serialized)
    for i in range(n_shards):
        writers[i].close()

def main():
    # todo find names
    sample_names = []
    for i, j, k in os.walk(IMG_DATA_PATH):
        for file in k:
            if file.startswith("BRATS") and file.endswith('.gz'):
                sample_names.append(file)
    sample_names = np.array(sample_names)
    np.random.shuffle(sample_names)
    val_sample_names = sample_names[:39]
    train_sample_names = sample_names[39:]
    # todo split train-set as 9:1 for training and validation
    preprocessing(os.path.join(TF_DATA_PATH, "val.tfrecords"), val_sample_names)
    preprocessing(os.path.join(TF_DATA_PATH, "train.tfrecords"), train_sample_names)

if __name__ == '__main__':
    main()

#END: OWN CODE
