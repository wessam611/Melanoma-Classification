AUTO = tf.data.experimental.AUTOTUNE

# Configuration
IMAGE_SIZE = [256, 256]
EPOCHS = 7
BATCH_SIZE = max(32 * strategy.num_replicas_in_sync, 64)

TF_RECORDS_FILES = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')
# TF_RECORDS_FILES = KaggleDatasets().get_gcs_path('melanoma-%ix%i'%(256,256))
# TF_RECORDS_FILES = "../input/siim-isic-melanoma-classification"
TRAIN_CSV = "../input/siim-isic-melanoma-classification/train.csv"
TEST_CSV = "../input/siim-isic-melanoma-classification/test.csv"
SUBMISSION_CSV = "submission.csv"

# Learning rate scheduler config
# Learning rate schedule for TPU, GPU and CPU.
# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.000005
LR_MAX = 0.0004
LR_MIN = 0.000001
LR_RAMPUP_EPOCHS = 1
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.75
