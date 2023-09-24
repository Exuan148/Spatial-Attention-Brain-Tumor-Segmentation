#START: OWN CODE
import tensorflow as tf
from config.u_net3d_config import *
def create_input(data_path):
    tfrecords_files = tf.io.gfile.glob(data_path)
    raw_image_dataset=tf.data.TFRecordDataset(tfrecords_files)
    def parse_tf_img(example_proto):
        image_feature_description = {
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_example = tf.io.parse_single_example(example_proto, image_feature_description)
        image = parsed_example['image_raw']  
        image = tf.compat.v1.decode_raw(image, tf.float32)  #  encode to the type of float32
        image = tf.reshape(image, [DIM1, DIM2, DIM3,CHANNEL ])

        image = tf.image.per_image_standardization(image)  # standardization
        # image = tf.cast(image, tf.float32)
        label=parsed_example['label']
        label = tf.compat.v1.decode_raw(label, tf.float32)  # encode to the type of float32
        label = tf.reshape(label, [DIM1, DIM2, DIM3, CLASS_NUM])
        # label=tf.cast(label,tf.float32)
        return image,label

    train_dataset=raw_image_dataset.map(parse_tf_img,num_parallel_calls=4)
    return train_dataset
	
#END: OWN CODE