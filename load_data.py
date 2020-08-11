from config import *
from data_augment import *

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

#     "age": tf.io.FixedLenFeature([], tf.float32),
#     "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.string),
#     "diagnosis": tf.io.FixedLenFeature([], tf.string), 
#     "benign_malignant": tf.io.FixedLenFeature([], tf.string),
 


feature_description = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64)
}


def parse_example(ex):
    return tf.io.parse_single_example(ex, feature_description)

def decode_image(W_LABEL, IMAGE_SIZE):
    def function(ex):
        image = ex['image']
        if W_LABEL:
            label = ex['target']
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
        ex['image'] = image
        if W_LABEL:
            return image, label
        return image
    return function

class DataLoader:

    def __init__(self, IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE,
                CACHE=True, W_LABELS=False, SPLIT=1.0, FROM_END=False):

        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TFR_FILES = TFR_FILES
        self.CSV_FILE = CSV_FILE
        self.CACHE = CACHE
        self.W_LABELS = W_LABELS
        self.SPLIT = SPLIT
        self.FROM_END = FROM_END
        

        self.load_csv()
        self.create_ds()

    def load_csv(self):
        self.csv_df = pd.read_csv(self.CSV_FILE)
        self.csv_size = self.csv_df.size
        self.datasize = self.csv_size
    
    def create_ds(self):
        ignore_order = tf.data.Options()
        if self.W_LABELS:
            ignore_order.experimental_deterministic = False
        
        self.dataset = tf.data.TFRecordDataset(self.TFR_FILES, num_parallel_reads = AUTO)
        self.dataset = self.dataset.with_options(ignore_order)
        self.dataset = self.dataset.map(parse_example)
        if self.CACHE:
            self.dataset = self.dataset.cache()
        self.dataset = self.dataset.map(decode_image(self.W_LABELS, self.IMAGE_SIZE))
        
        if self.W_LABELS:
            self.dataset = self.dataset.repeat()
            self.dataset.map(transform)
        
        self.dataset = self.dataset.prefetch(AUTO)
        self.dataset = self.dataset.shuffle(256)
        self.dataset = self.dataset.batch(self.BATCH_SIZE, self.W_LABELS)
        
    def get_dataset(self):
        return self.dataset
    
    def get_iterations(self):
        return self.datasize // self.BATCH_SIZE
        
class TrainDataLoader(DataLoader):
    """
    Also used for validation (by setting VAL=True)
    """
    def __init__(self, IMAGE_SIZE, BATCH_SIZE, VAL=False, CACHE=True):
        TFR_FILES = tf.io.gfile.glob(TF_RECORDS_FILES + '/tfrecords/train*.tfrec')
        if VAL:
            TFR_FILES = TFR_FILES[-2:]
        else:
            TFR_FILES = TFR_FILES[0:-2]
        print(TFR_FILES)
        super(TrainDataLoader, self).__init__( IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE=TRAIN_CSV,
                CACHE=CACHE, W_LABELS=True)

class TestDataLoader(DataLoader):
    def __init__(self, IMAGE_SIZE, BATCH_SIZE):
        TFR_FILES = tf.io.gfile.glob(TF_RECORDS_FILES + '/tfrecords/test*.tfrec')
        super(TrainDataLoader, self).__init__(IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE=TEST_CSV,
                CACHE=False, W_LABELS=False, SPLIT=1.0, FROM_END=False)