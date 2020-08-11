AUTO = tf.data.experimental.AUTOTUNE

# Configuration
IMAGE_SIZE = [224, 224]
EPOCHS = 5
FOLDS = 3
SEED = 777
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

TF_RECORDS_FILES = "../input/siim-isic-melanoma-classification/tfrecords"
TRAIN_CSV = "../input/siim-isic-melanoma-classification/train.csv"
TEST_CSV = "../input/siim-isic-melanoma-classification/test.csv"
SUBMISSION_CSV = "submission.csv"