# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from config import *
from load_data import *

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

from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import cv2

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


train_dataset = TrainDataLoader(IMAGE_SIZE, BATCH_SIZE, VAL=False, CACHE=True)
val_dataset = TrainDataLoader(IMAGE_SIZE, BATCH_SIZE, VAL=True, CACHE=True)

with strategy.scope():    
    pre1 = tf.keras.applications.ResNet101(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
    pre1.trainable=True
    model1 = tf.keras.Sequential([
        pre1,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, 'sigmoid')#, kernel_regularizer='l2')
    ])
    model1.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

model1.fit(train_dataset.get_dataset(), 
           steps_per_epoch=5,#1811,#train_dataset.get_iterations(), 
           epochs=EPOCHS,
          callbacks = [lr_callback],
          validation_data=val_dataset.get_dataset(),
          validation_steps=5)#258)