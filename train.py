# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import gc
import random, re, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.losses import Loss
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import efficientnet.tfkeras as efn

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

from kaggle_datasets import KaggleDatasets

import cv2

from config import *
from load_data import *

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def train():
    pres = [
            efn.EfficientNetB5,
            efn.EfficientNetB6,
            efn.EfficientNetB3,
            efn.EfficientNetB4,
            efn.EfficientNetB4,
            efn.EfficientNetB4
    ]
    dropouts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    models = []
    hists = []
    TFR_FILES = np.asarray(tf.io.gfile.glob(TF_RECORDS_FILES + '/train*.tfrec'))
    kf = KFold(n_splits=6)
    i = 0
    for train, val in kf.split(TFR_FILES):
        print(pres[i], dropouts[i])
        train = TFR_FILES[train]
        val = TFR_FILES[val]
        train_dataset = TrainDataLoader(IMAGE_SIZE, BATCH_SIZE, False, True, train)
        val_dataset = TrainDataLoader(IMAGE_SIZE, BATCH_SIZE, True, True, val)
        with strategy.scope():
            pre = pres[i](weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
            pre.trainable=True
            model = tf.keras.Sequential([
                pre,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(dropouts[i]),
                tf.keras.layers.Dense(32, 'relu')
            ])
            in_img = tf.keras.layers.Input((*IMAGE_SIZE, 3), name='in_img')
            in_feats = tf.keras.layers.Input((8), name='in_feats')
            y_img = model(in_img)
            y_feats = tf.keras.layers.Dense(32, 'relu')(in_feats)
            y = tf.keras.layers.Concatenate(axis=1)([y_img, y_feats])
            y = tf.keras.layers.Dropout(dropouts[i]/2)(y)
            y = tf.keras.layers.Dense(1)(y)
            y = tf.keras.layers.Activation('sigmoid')(y)
            model = Model(inputs=[in_img, in_feats], outputs=y)
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
            model.compile('adam', loss=loss, metrics=[ 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
                get_f1, 'AUC'
            ])
        hist = model.fit(train_dataset.get_dataset(), 
            steps_per_epoch=202,#train_dataset.get_iterations(), 
            epochs=EPOCHS,
            callbacks = [lr_callback],
            validation_data=val_dataset.get_dataset(),
            validation_steps=34,
                verbose=0)
        print('train_auc', hist.history['accuracy'])
        print('val_auc', hist.history['val_accuracy'])
        print('train_acc', hist.history['auc'])
        print('val_acc', hist.history['val_auc'])
        models.append(model)
        hists.append(hist)
        i += 1
        if i == 5:
            break

    aucs, models = zip(*sorted(zip([hist.history['val_auc'][-1] for hist in hists], models)))
    models = list(models)
    aucs = list(aucs)
    # models = models[2:]

    print(aucs)

    with strategy.scope():
        in_img = tf.keras.layers.Input((*IMAGE_SIZE, 3), name='in_img')
        in_feats = tf.keras.layers.Input((8), name='in_feats')
        preds = []
        for model in models:
            model.trainable = False
            temp_mod = Model(model.input, model.layers[-2].output)
            preds.append(temp_mod({'in_img': in_img, 'in_feats': in_feats}))
        y = tf.keras.layers.Concatenate(axis=1)(preds)
        y = tf.keras.layers.Dense(1, 'sigmoid')(y)
        forest = Model([in_img, in_feats], y)
        
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)
        forest.compile('adam', loss=loss, metrics=[ 
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
            get_f1, 'AUC'
        ])
    train_dataset = TrainDataLoader(IMAGE_SIZE, BATCH_SIZE//len(models), False, True, TFR_FILES) # with augmentation
    forest.fit(train_dataset.get_dataset(), 
            steps_per_epoch=236*len(models),#train_dataset.get_iterations(), 
            epochs=2,
            callbacks = [lr_callback]
            )
    return forest