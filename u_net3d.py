
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv3DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv3D, MaxPooling3D,UpSampling3D
from tensorflow.keras.layers import Dense
from config.u_net3d_config import *


#START: OWN CODE
class U_Net():
    def __init__(self):
        self.dim1 = DIM1
        self.dim2 = DIM2
        self.dim3 = DIM3
        self.channels = CHANNEL
        self.shape = (self.dim1,self.dim2, self.dim3, self.channels)
        self.class_num=CLASS_NUM
        self.weight_loss=WEIGHT_LOSS
        self.batch_norm = BATCH_NORM
        self.ATTENTION_BLOCK=ATTENTION_BLOCK
        self.PROJECT_EXCITE_BLOCK=PROJECT_EXCITE_BLOCK
        # self.with_attention
        optimizer = OPTIMIZER
        self.unet = self.build_unet()  
        self.unet.compile(optimizer=optimizer, loss=[self.dice_loss],
                          metrics=[self.dice_coef])
        self.unet.summary()

    def project_excite_block(self,input_tensor):
        if PROJECT_EXCITE_BLOCK == False:
            return input_tensor
        shape = [input_tensor.get_shape()[1], input_tensor.get_shape()[2], input_tensor.get_shape()[3]]
        n_channel = input_tensor.get_shape()[4]
        x1 = tf.nn.avg_pool(input_tensor, ksize=[1, shape[0], shape[1],1, 1], strides=[1, shape[0], shape[1],1, 1], padding='SAME')
        x1 = tf.tile(x1, (1,shape[0], shape[1], 1, 1))
        x2 = tf.nn.avg_pool(input_tensor, ksize=[1, shape[0], 1,shape[2], 1], strides=[1, shape[0], 1,shape[2], 1], padding='SAME')
        x2 = tf.tile(x2, (1,shape[0], 1, shape[2], 1))
        x3 = tf.nn.avg_pool(input_tensor, ksize=[1, 1, shape[1],shape[2], 1], strides=[1, 1, shape[1],shape[2], 1], padding='SAME')
        x3 = tf.tile(x3, (1,1, shape[1], shape[2], 1))
        x_add = x1 + x2 + x3
        x_dense1 = Dense(n_channel // 4, 'relu')(x_add)
        x_dense2 = Dense(n_channel, 'sigmoid')(x_dense1)
        X = input_tensor * x_dense2
        return X

    def attention_block(self,input_tensor, kernel_size=3, padding='same'):
        if ATTENTION_BLOCK == False:
            return input_tensor
        n_filters = int((input_tensor.get_shape()[-1] + 1) // 2)
        x1 = Conv3D(n_filters, kernel_size, activation='relu', padding=padding)(input_tensor)
        x2 = Conv3D(1, kernel_size, activation='sigmoid', padding=padding)(x1)

        x3 = input_tensor * x2
        X = input_tensor + x3
        return X

#END: OWN CODE

    def build_unet(self, dropout=0.1,padding='same'):

        x = Input(shape=self.shape)

        conv1 = Conv3D(8, 3, activation='relu', padding='same', data_format="channels_last")(x)
		#START: OWN CODE
        if self.batch_norm:
            conv1 = BatchNormalization()(conv1)
        conv1 = Conv3D(8, 3, activation='relu', padding='same')(conv1)
        if self.batch_norm:
            conv1 = BatchNormalization()(conv1)
		#END: OWN CODE
        pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
        pool1=self.project_excite_block(pool1)
        conv2 = Conv3D(16, 3, activation='relu', padding='same')(pool1)
		#START: OWN CODE
        if self.batch_norm:
            conv2 = BatchNormalization()(conv2)
        conv2 = Conv3D(16, 3, activation='relu', padding='same')(conv2)
        if self.batch_norm:
            conv2 = BatchNormalization()(conv2)
		#END: OWN CODE
        pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
        pool2 = self.project_excite_block(pool2)
        conv3 = Conv3D(32, 3, activation='relu', padding='same')(pool2)
		#START: OWN CODE
        if self.batch_norm:
            conv3 = BatchNormalization()(conv3)
        conv3 = Conv3D(32, 3, activation='relu', padding='same')(conv3)
        if self.batch_norm:
            conv3 = BatchNormalization()(conv3)
		#END: OWN CODE
        pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
        pool3 = self.project_excite_block(pool3)
        conv4 = Conv3D(64, 3, activation='relu', padding='same')(pool3)
		#START: OWN CODE
        if self.batch_norm:
            conv4 = BatchNormalization()(conv4)
        conv4 = Conv3D(64, 3, activation='relu', padding='same')(conv4)
        if self.batch_norm:
            conv4 = BatchNormalization()(conv4)
		#END: OWN CODE
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)
        pool4=self.attention_block(pool4)
        pool4 = self.project_excite_block(pool4)
        conv5 = Conv3D(128, 3, activation='relu', padding='same')(pool4)
		#START: OWN CODE
        if self.batch_norm:
            conv5 = BatchNormalization()(conv5)
        conv5 = Conv3D(128, 3, activation='relu', padding='same')(conv5)
        if self.batch_norm:
            conv5 = BatchNormalization()(conv5)
		#END: OWN CODE
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv3D(64, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=-1)
        merge6=self.attention_block(merge6)
        merge6 = self.project_excite_block(merge6)
        conv6 = Conv3D(64, 3, activation='relu', padding='same')(merge6)
		#START: OWN CODE
        if self.batch_norm:
            conv6 = BatchNormalization()(conv6)
        conv6 = Conv3D(64, 3, activation='relu', padding='same')(conv6)
        if self.batch_norm:
            conv6 = BatchNormalization()(conv6)
		#END: OWN CODE
        up7 = Conv3D(32, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=-1)
        merge7 = self.attention_block(merge7)
        merge7 = self.project_excite_block(merge7)
        conv7 = Conv3D(32, 3, activation='relu', padding='same')(merge7)
		#START: OWN CODE
        if self.batch_norm:
            conv7 = BatchNormalization()(conv7)
        conv7 = Conv3D(32, 3, activation='relu', padding='same')(conv7)
        if self.batch_norm:
            conv7 = BatchNormalization()(conv7)
		#END: OWN CODE
        up8 = Conv3D(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=-1)
        merge8 = self.attention_block(merge8)
        merge8 = self.project_excite_block(merge8)
        conv8 = Conv3D(16, 3, activation='relu', padding='same')(merge8)
		#START: OWN CODE
        if self.batch_norm:
            conv8 = BatchNormalization()(conv8)
        conv8 = Conv3D(16, 3, activation='relu', padding='same')(conv8)
        if self.batch_norm:
            conv8 = BatchNormalization()(conv8)
		#END: OWN CODE
        up9 = Conv3D(8, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=-1)
        merge9 = self.attention_block(merge9)
        merge9 = self.project_excite_block(merge9)
        conv9 = Conv3D(8, 3, activation='relu', padding='same')(merge9)
		#START: OWN CODE
        if self.batch_norm:
            conv9 = BatchNormalization()(conv9)
        conv9 = Conv3D(8, 3, activation='relu', padding='same')(conv9)
        if self.batch_norm:
            conv9 = BatchNormalization()(conv9)
		#END: OWN CODE
        output = Conv3D(self.class_num, 1, activation='sigmoid')(conv9)

        return Model(x, output)

    def dice_loss(self, y_true, y_pred):
        """
        multi label dice loss with weighted
        Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                           self.numclass],Y_pred is softmax result
        Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                           self.numclass],Y_gt is one hot result
        weight_loss: numpy array of shape (C,) where C is the number of classes,eg:[0,1,1,1]
        :return:
        """
        weight_loss = np.array(self.weight_loss)
        smooth = 1.e-5
        smooth_tf = tf.constant(smooth, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        intersection = y_pred * y_true
        intersection = tf.reduce_sum(intersection, axis=(1, 2, 3))
        union = y_pred + y_true
        union = tf.reduce_sum(union, axis=(1, 2, 3))
        dice_coefs = tf.reduce_mean(2. * (intersection + smooth_tf) / (union + smooth_tf), axis=0)
        return 1-tf.reduce_mean(weight_loss * dice_coefs)

    def dice_coef(self, y_true, y_pred):
        """
        multi label dice loss with weighted
        Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                           self.numclass],Y_pred is softmax result
        Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                           self.numclass],Y_gt is one hot result
        weight_loss: numpy array of shape (C,) where C is the number of classes,eg:[0,1,1,1]
        :return:
        """
        weight_loss = np.array(self.weight_loss)
        smooth = 1.e-5
        smooth_tf = tf.constant(smooth, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        intersection = y_pred * y_true
        intersection = tf.reduce_sum(intersection, axis=(1, 2, 3))
        union = y_pred + y_true
        union = tf.reduce_sum(union, axis=(1, 2, 3))
        dice_coefs = tf.reduce_mean(2. * (intersection + smooth_tf) / (union + smooth_tf), axis=0)
        return tf.reduce_mean(weight_loss * dice_coefs)

   


