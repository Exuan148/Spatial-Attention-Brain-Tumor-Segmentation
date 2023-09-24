#START: OWN CODE
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from inputs import create_input
import pynvml
from u_net3d import U_Net
import os
import datetime
from config.u_net3d_config import *
# todo fetch GPU
pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpulist = []
for i in range(deviceCount):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 指定显卡号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if (meminfo.used / (1024 ** 2) <= 315.4375):
        print("GPU:{i} is available".format(i=i))
        gpulist.append(i)
        if(len(gpulist)==GPU_MAX_NUM):
            break
if len(gpulist) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for i in range(len(gpulist) - 1):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] + str(gpulist[i]) + ","
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] + str(gpulist[-1])
print(os.environ["CUDA_VISIBLE_DEVICES"])



def trainer():
    mirrored_strategy = tf.distribute.MirroredStrategy(DISTRIIBUTED_STRATEGY_GPUS)
    with mirrored_strategy.scope():
        # todo load TFRecord data
        train_dataset = create_input(TRAIN_DATA_PATH)
        train_batched = train_dataset.batch(batch_size=BATCH_SIZE)

        val_dataset = create_input(VAL_DATA_PATH)
        val_batched = val_dataset.batch(batch_size=BATCH_SIZE)

        # todo initialize model
        model = U_Net()

        # todo load checkpoints
        checkpoint_names = []
        initial_epoch_of_training = 0
        for i, j, k in os.walk(CHECKPOINTS_PATH):
            for file in k:
                if file.endswith('.h5'):
                    checkpoint_names.append(file)
        print(len(checkpoint_names), "-----------")
        if len(checkpoint_names) > 0:

            max_epoch = 0
            for checkpoint_name in checkpoint_names:
                if max_epoch < int(checkpoint_name.replace('.h5', '').split('_')[-1]):
                    max_epoch = int(checkpoint_name.replace('.h5', '').split('_')[-1])
                    name = checkpoint_name
            initial_epoch_of_training = max_epoch
            model.unet.load_weights(os.path.join(CHECKPOINTS_PATH, name))

        # todo callbacks
        # TensorBoard
        logdir = os.path.join(LOG_PATH, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        # csv_logger
        os.makedirs(os.path.dirname(TRAINING_CSV), exist_ok=True)
        csv_logger = tf.keras.callbacks.CSVLogger(TRAINING_CSV + 'BrainTumor.csv', append=True)
        # Model-checkpoings
        path = CHECKPOINTS_PATH
        Model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path,
                                                                                  "BrainTumor_{loss:.2f}_{val_loss:.2f}_{epoch}.h5"),
                                                            save_best_only=False,
                                                            save_weights_only=True,
                                                            verbose=1,
                                                            period=10)
        callbacks = [
            # EarlyStopping(patience=100, verbose=1),
            tensorboard_callback,
            csv_logger,
            # ReduceLROnPlateau(factor=0.5, patience=15, min_lr=0.00005, verbose=1),
            Model_callback]

        results = model.unet.fit(train_batched,
                                 validation_data=val_batched,
                                 validation_steps=VALIDATION_STEP,
                                 initial_epoch=initial_epoch_of_training,
                                 batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                                 callbacks=callbacks, shuffle=True)


if __name__ == '__main__':
    trainer()

#END: OWN CODE