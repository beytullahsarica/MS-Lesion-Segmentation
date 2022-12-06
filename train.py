# -*- coding: utf-8 -*-
import argparse
import tensorflow
import gc
import tensorflow.keras.backend as K
import numpy as np
import os
import resource
import segmentation_models as sm
import time
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import model_from_json
from random import seed
from sklearn.model_selection import train_test_split
from time import strftime
import gdown

from augmentation import get_training_augmentation, get_validation_augmentation, get_preprocessing
from dataloader import DataLoader
from dataset import MSDataset

sm.set_framework('tf.keras')
sm.framework()
tensorflow.keras.backend.set_image_data_format('channels_last')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tensorflow.get_logger().setLevel('INFO')

seed(2022)
width = 224
height = 224
img_channel = 3
mask_channel = 1


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("ClearMemory on_epoch_end....")
        print("resource usages: ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        gc.collect()
        tensorflow.keras.backend.clear_session()


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_out = true_positives / (possible_positives + K.epsilon())
    return recall_out


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_out = true_positives / (predicted_positives + K.epsilon())
    return precision_out


def get_optim_loss_metrics(lr=0.0001, n_classes=1):
    optim = tensorflow.keras.optimizers.Adam(lr)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5), precision, recall]
    return optim, total_loss, metrics


def read_model_from_json(model_name):
    model_path = os.path.join("models", model_name + ".json")
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    return model_from_json(model_json)


def get_model(model_name="isbi_dense_res_u_net_ag_eca_aspp", lr=0.0001, n_classes=1):
    model = read_model_from_json(model_name=model_name)
    optim, total_loss, metrics = get_optim_loss_metrics(lr=lr, n_classes=n_classes)
    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
    return model


def get_data_from_npy_files(isbi=True, validation_split_size_percentage=0.10, preprocess_input=None):
    if isbi is False:
        merged_train_images = np.load("dataset/msseg2016/train_images_224.npy").astype(np.float32)
        merged_train_masks = np.load("dataset/msseg2016/train_masks_224.npy").astype(np.float32)
        merged_train_images = merged_train_images.reshape((len(merged_train_images), width, height, img_channel))
        merged_train_masks = merged_train_masks.reshape((len(merged_train_masks), width, height, mask_channel))
    else:
        rater1_train_images = np.load("dataset/isbi2015/rater1_images_224.npy").astype(np.float32)
        rater1_train_masks = np.load("dataset/isbi2015/rater1_masks_224.npy").astype(np.float32)

        rater2_train_images = np.load("dataset/isbi2015/rater2_images_224.npy").astype(np.float32)
        rater2_train_masks = np.load("dataset/isbi2015/rater2_masks_224.npy").astype(np.float32)

        # convert to tensor format 4D(batch_size, width, height,channel)
        rater1_train_images = rater1_train_images.reshape((len(rater1_train_images), width, height, img_channel))
        rater1_train_masks = rater1_train_masks.reshape((len(rater1_train_masks), width, height, mask_channel))
        rater2_train_images = rater2_train_images.reshape((len(rater2_train_images), width, height, img_channel))
        rater2_train_masks = rater2_train_masks.reshape((len(rater2_train_masks), width, height, mask_channel))
        # concatenate two raters data into the one
        merged_train_images = np.concatenate((rater1_train_images, rater2_train_images))
        merged_train_masks = np.concatenate((rater1_train_masks, rater2_train_masks))

    train_images, valid_images, train_masks, valid_masks = train_test_split(merged_train_images,
                                                                            merged_train_masks,
                                                                            test_size=validation_split_size_percentage,
                                                                            random_state=2022)
    train_images = preprocess_input(train_images)
    train_masks = preprocess_input(train_masks)
    valid_images = preprocess_input(valid_images)
    valid_masks = preprocess_input(valid_masks)

    return train_images, train_masks, valid_images, valid_masks


def get_datasets_and_dataloaders(preprocess_input=None, batch_size=8, isbi=True, validation_split_size_percentage=0.10):
    train_images, train_masks, valid_images, valid_masks = get_data_from_npy_files(isbi=isbi,
                                                                                   validation_split_size_percentage=validation_split_size_percentage,
                                                                                   preprocess_input=preprocess_input)

    print(f"train images: {len(train_images)} - train_masks: {len(train_masks)}")
    print(f"valid images: {len(valid_images)} - valid_masks: {len(valid_masks)}")

    train_dataset = MSDataset(train_images=train_images,
                              train_labels=train_masks,
                              augmentation=get_training_augmentation(width=width, height=height),
                              preprocessing=get_preprocessing(preprocess_input))

    valid_dataset = MSDataset(train_images=valid_images,
                              train_labels=valid_masks,
                              augmentation=get_validation_augmentation(width=width, height=height),
                              preprocessing=get_preprocessing(preprocess_input))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    return train_dataset, valid_dataset, train_dataloader, valid_dataloader


def get_callbacks(tensorboard_callback, best_modal_name):
    saved_weight_path = os.path.join('training_output', best_modal_name)
    print(f"best model weights path: {saved_weight_path}")
    return [
        ClearMemory(),
        tensorboard_callback,
        tensorflow.keras.callbacks.ModelCheckpoint(saved_weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
        tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, cooldown=10, min_lr=1e-5),
        tensorflow.keras.callbacks.EarlyStopping(patience=50, verbose=1, monitor='val_loss', mode="min")
    ]


def start_train(dataset="isbi2015", model_name="isbi_dense_res_u_net_ag_eca_aspp", batch_size=8, epochs=10, lr=0.0001, rater='rater12',
                root_path='MS-Lesion-Segmentation',
                preprocess_input=None, n_classes=1, fold_name_id=1, modality='all', validation_split_size_percentage=0.10):
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = get_datasets_and_dataloaders(preprocess_input=preprocess_input,
                                                                                                    batch_size=batch_size,
                                                                                                    isbi=("isbi2015" == dataset),
                                                                                                    validation_split_size_percentage=validation_split_size_percentage)
    model = get_model(model_name=model_name, lr=lr, n_classes=n_classes)
    if "msseg2016" == dataset:
        print(f"pre-train from isbi2015 to msseg2016")
        pt_share_link = "https://drive.google.com/file/d/14_GdCNHIwKpgj6GFuh0gEkQmdCxCbDtz/view?usp=sharing"
        pt_id = pt_share_link.split("/")[-2]
        url_to_drive = f"https://drive.google.com/uc?id={pt_id}&confirm=t"
        model_checkpoint = "isbi_dense_res_u_net_ag_eca_aspp.h5"
        gdown.download(url_to_drive, model_checkpoint, quiet=False)
        model.load_weights(model_checkpoint)

    best_modal_name = f"{dataset}/{rater}_{modality}_{model_name}_fold_{str(fold_name_id)}_{str(width)}_best_model_all.h5"
    model_history = f"{dataset}/{rater}_{modality}_{model_name}_fold_{str(fold_name_id)}_{str(width)}_model_history.npy"

    import datetime
    tensorboard_log_path = os.path.join(root_path, "training_output", dataset, "logs", model_name)
    if not os.path.exists(tensorboard_log_path):
        os.makedirs(tensorboard_log_path)

    log_dir = os.path.join(tensorboard_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    time_start = time.time()
    print("start time at {}".format(strftime("%m/%d/%Y, %H:%M:%S", time.gmtime(time_start))))
    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(valid_dataset)}")
    print(f"rater: {rater}")
    print(f"model type: {model_name}")
    print(f"modality: {modality}")
    print(f"fold_name_id: {fold_name_id}")
    print(f"using dataloader...")
    history = model.fit(train_dataloader,
                        steps_per_epoch=len(train_dataloader),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=get_callbacks(tensorboard_callback=tensorboard_callback, best_modal_name=best_modal_name),
                        validation_data=valid_dataloader,
                        validation_steps=len(valid_dataloader),
                        verbose=1)

    end_time = time.time()
    print("end time at {}".format(strftime("%m/%d/%Y, %H:%M:%S", time.gmtime(end_time))))
    # save history for further usages such as plotting acc and other metrics
    np.save(os.path.join(root_path, 'training_output', model_history), history.history)


def check_version():
    print("tensorflow version: ", tensorflow.__version__)
    print("keras version: ", tensorflow.keras.__version__)


def check_device():
    device_name = tensorflow.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))


def main():
    root_directory = os.path.join("MS-Lesion-Segmentation")

    parser = argparse.ArgumentParser(description='Training Dense Residual network for MS lesion segmentation')
    parser.add_argument('--dataset', type=str, help='--dataset "isbi2015" ', default="isbi2015")
    parser.add_argument('--model_name', type=str, help='--model_name "dense_res_u_net_ag_eca_aspp" ', default="dense_res_u_net_ag_eca_aspp")
    parser.add_argument('--modality', type=str, help='--modality "all" ', default="all")
    parser.add_argument('--rater', type=str, help='--rater "rater12" ', default="rater12")
    parser.add_argument('--validation_split_percentage', type=float, help='--validation_split_percentage 0.10',
                        default=0.10)
    parser.add_argument('--batch_size', type=int, help='--batch_size 8 default size is 8', default=8)
    parser.add_argument('--epochs', type=int, help='--epochs 50 default epochs is 10', default=10)
    parser.add_argument('--fold_name_id', type=int, help='--fold_name_id 1 default epochs is 1', default=1)
    parser.add_argument('--lr', type=float, help='--lr 0.0001 default learning rate is 0.0001', default=1e-4)

    args = parser.parse_args()
    trained_model_name = args.model_name
    print("dataset: ", args.dataset)
    print("rater: ", args.rater)
    print("trained_model_name: ", trained_model_name)
    print("validation_split_percentage: ", args.validation_split_percentage)
    print("batch_size: ", args.batch_size)
    print("epochs: ", args.epochs)
    print("lr (learning rate): ", args.lr)

    check_version()
    check_device()
    preprocess_input = sm.get_preprocessing("resnet34")

    start_train(dataset=args.dataset,
                model_name=trained_model_name,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                rater=args.rater,
                root_path=root_directory,
                preprocess_input=preprocess_input,
                n_classes=1,
                fold_name_id=args.fold_name_id,
                modality=args.modality,
                validation_split_size_percentage=args.validation_split_percentage)


if __name__ == "__main__":
    main()
