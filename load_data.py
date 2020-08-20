import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd

from config import *
from data_augment import *

#     "age": tf.io.FixedLenFeature([], tf.float32),
#     "anatom_site_general_challenge": tf.io.FixedLenFeature([], tf.string),
#     "diagnosis": tf.io.FixedLenFeature([], tf.string), 
#     "benign_malignant": tf.io.FixedLenFeature([], tf.string),
 


train_feature_description = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
#     'sex': tf.io.FixedLenFeature([], tf.int64),
    'age_approx': tf.io.FixedLenFeature([], tf.int64),
    'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    "target": tf.io.FixedLenFeature([], tf.int64)
}

test_feature_description = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
#     'sex': tf.io.FixedLenFeature([], tf.int64),
    'age_approx': tf.io.FixedLenFeature([], tf.int64),
    'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
}


def parse_example(W_LABEL):
    def function(ex):
        if W_LABEL:
            feature_description = train_feature_description
        else:
            feature_description = test_feature_description
        return tf.io.parse_single_example(ex, feature_description)
    return function

def decode_image(W_LABEL, IMAGE_SIZE):
    def function(ex):
        image = ex['image']
        age = (ex['age_approx']-51)/(20)
#         source = tf.cast(tf.one_hot(ex['source'], 3), tf.float32)
        ana = tf.cast(tf.one_hot(ex['anatom_site_general_challenge'], 7), tf.float32)
        feats = tf.concat([ana, tf.cast(tf.expand_dims(age, 0), tf.float32)], 0)
        if W_LABEL:
            label = ex['target']
            feature_description = train_feature_description
        else:
            feature_description = test_feature_description
            image_name = ex['image_name']
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
        ex['image'] = image
        if W_LABEL:
            return image, feats, label
        return image, feats, image_name
    return function

def increase_pos(images, feats, labels):
    LEN = tf.size(labels)
    pos = tf.squeeze(tf.where(tf.equal(labels, 1)), -1)
    neg = tf.squeeze(tf.where(tf.equal(labels, 0)), -1)
    if tf.size(pos) == 0:
        return images, labels
    if tf.size(neg) == 0:
        return images, labels
    neg = tf.tile(neg, [1+int(((LEN)//(2*tf.shape(neg)[0])))], name='neg_tile')
    neg = neg[0:LEN//2]
    pos = tf.tile(pos, multiples=[1 + (LEN)//(2*tf.size(pos))], name='pos_tile')
    pos = pos[0:LEN//2]
    indices = tf.concat([pos, neg], 0)
    indices = tf.random.shuffle(indices)
    imgs = tf.gather(images, indices)
    lbls = tf.gather(labels, indices)
    fts = tf.gather(feats, indices)
    return imgs, fts, lbls

def input_to_dict(images, feats, labels):
    
    return {'in_img': images, 'in_feats': feats}, labels


class DataLoader:

    def __init__(self, IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE,
                CACHE=True, W_LABELS=False, SPLIT=1.0, FROM_END=False, VAL=False):

        self.IMAGE_SIZE = IMAGE_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.TFR_FILES = TFR_FILES
        self.CSV_FILE = CSV_FILE
        self.CACHE = CACHE
        self.W_LABELS = W_LABELS
        self.SPLIT = SPLIT
        self.FROM_END = FROM_END
        self.VAL = VAL
        

        self.load_csv()
        self.create_ds()

    def load_csv(self):
        pass
#         self.csv_df = pd.read_csv(self.CSV_FILE)
#         self.csv_size = self.csv_df.size
#         self.datasize = self.csv_size
    
    def create_ds(self):
        ignore_order = tf.data.Options()
        if self.W_LABELS:
            ignore_order.experimental_deterministic = False
        
        self.dataset = tf.data.TFRecordDataset(self.TFR_FILES, num_parallel_reads = AUTO)
        if self.CACHE:
#             self.dataset = self.dataset.cache()
            self.dataset = self.dataset.with_options(ignore_order)
        self.dataset = self.dataset.map(parse_example(self.W_LABELS), num_parallel_calls=AUTO)
        
        self.dataset = self.dataset.map(decode_image(self.W_LABELS, self.IMAGE_SIZE), num_parallel_calls=AUTO)
        if self.W_LABELS:
            self.dataset = self.dataset.repeat()
#             self.dataset = self.dataset.batch(2048)
#             self.dataset = self.dataset.map(increase_pos, num_parallel_calls=AUTO)
#             self.dataset = self.dataset.unbatch()
            self.dataset = self.dataset.shuffle(1024)
        self.dataset = self.dataset.batch(self.BATCH_SIZE, self.W_LABELS)
        if self.W_LABELS and (not self.VAL):
            self.dataset = self.dataset.map(transform_batch, num_parallel_calls=AUTO)
#         self.dataset = self.dataset.map(merge_feat, num_parallel_calls=AUTO)
        self.dataset = self.dataset.map(input_to_dict, num_parallel_calls=AUTO)
        self.dataset = self.dataset.prefetch(AUTO)
        
    def get_dataset(self):
        return self.dataset
    
    def get_iterations(self):
        return self.datasize // self.BATCH_SIZE
        
class TrainDataLoader(DataLoader):
    """
    Also used for validation (by setting VAL=True)
    """
    def __init__(self, IMAGE_SIZE, BATCH_SIZE, VAL=False, CACHE=True, TFR_FILES=None):
        if TFR_FILES is None:
            TFR_FILES = tf.io.gfile.glob(TF_RECORDS_FILES + '/train*.tfrec')
            if VAL:
                TFR_FILES = TFR_FILES[-5:]
            else:
                TFR_FILES = TFR_FILES[0:-5]
        super(TrainDataLoader, self).__init__( IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE=TRAIN_CSV,
                CACHE=CACHE, W_LABELS=True, VAL= VAL)

class TestDataLoader(DataLoader):
    def __init__(self, IMAGE_SIZE, BATCH_SIZE):
        TFR_FILES = tf.io.gfile.glob(TF_RECORDS_FILES + '/test*.tfrec')
        super(TestDataLoader, self).__init__(IMAGE_SIZE, BATCH_SIZE, TFR_FILES, CSV_FILE=TEST_CSV,
                CACHE=False, W_LABELS=False, SPLIT=1.0, FROM_END=False)