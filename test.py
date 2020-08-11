from config import *
from load_data import *

import os
import gc
import random, re, math

import numpy as np
import pandas as pd
import tensorflow as tf


def predict(model, save_model=True):
    test_dataset = TestDataLoader(IMAGE_SIZE, BATCH_SIZE)
    row_list = []

    for batch in test_dataset.get_dataset():
        res = model.predict_on_batch(batch[0])
        preds = [[img_name.decode("utf-8"), pred[0]] for img_name, pred in zip(batch[1].numpy(), res)]
        row_list = row_list + preds
    df = pd.DataFrame(row_list, 
                columns =['image_name', 'target']).sort_values(by='image_name')
    df.to_csv(SUBMISSION_CSV)
    if save_model:
        tf.saved_model.save(model, 'model.h5')