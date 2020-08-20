from config import *
from load_data import *

import os
import gc
import random, re, math

import numpy as np
import pandas as pd
import tensorflow as tf

from load_data import *


def pred_to_label(pred, hthresh=0.766, lthresh=0.233):
    if pred < lthresh:
        return 0.0
    elif pred > hthresh:
        return 1.0
    else:
        return pred


def predict(model, save_model=True, thresh=(0.7, 0.25), model_name='model'):
   
    test_dataset = TestDataLoader(IMAGE_SIZE, BATCH_SIZE)
    row_list = []
    row_list_threshed = []

    for batch in test_dataset.get_dataset():
        res = model.predict_on_batch(batch[0])
        preds = [[img_name.decode("utf-8"), pred[0]] for img_name, pred in zip(batch[1].numpy(), res)]
        row_list = row_list + preds

        preds_threshed = [[img_name.decode("utf-8"), pred_to_label(pred[0]), thresh[0], thresh[1]] for img_name, pred in zip(batch[1].numpy(), res)]
        row_list_threshed = row_list_threshed + preds_threshed
    df = pd.DataFrame(row_list, 
                columns =['image_name', 'target']).sort_values(by='image_name')
    df.to_csv(SUBMISSION_CSV, index=False)

    dft = pd.DataFrame(row_list_threshed, 
                columns =['image_name', 'target']).sort_values(by='image_name')
    dft.to_csv('t'+SUBMISSION_CSV, index=False)

    if save_model:
        model.save('{}.h5'.format(model_name))
    return df, dft