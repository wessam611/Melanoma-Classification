# [SIIM-ISIC Melanoma-Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification/) (Kaggle competition)

Image Classification Competition on Kaggle for detecting Melanoma skin Cancer (my best score: pu: 0.9235, pr: 0.9124) [notebook](https://www.kaggle.com/wessam611/siim-melanoma-classification?scriptVersionId=41079699)

## Experience.

* It's my first competition on Kaggle to take seriously. I've learned some about the kaggle community, enjoyed the experience of joing a live competition on kaggle while reading other's ideas from the discussion and notebooks section.
* I've gained knew knowledge through the competition including (AUC metric, EfficientNets and crossEntropy label smoothing).
* I've experimented more with Tensorflow Dataset module and TFRecords and feel more comfortable with it now.

## Experiments.
* I've tried using different models including (ResNets 'overfits', VGG, DenseNets, Inception 'overfits' and EfficientNets 'works best').
* After using EfficientNets I built a Ensemble model using weighted sum of 5 EfficientNets predictions (weighted sums didn't improve the result much from just using the average prediction).
* I've oversampled positive examples in the dataset in every 2048 records parallel to training (chech older versions of the notebook) making the distribution of the training data (50% pos, 50% neg) which resulted in overfitting (Same picture is used many times even with heavy augmentations).
* I've settled on using a more balanced [Dataset](https://www.kaggle.com/cdeotte/512x512-melanoma-tfrecords-70k-images) which increased the accuracy and auc of the model while preventing overfitting.

### Notes.
* I've had different problems with Memory and overfitting, so I'd really appreciate any advice reagrding my code and this competition in general.
* Follow me on [Kaggle](https://www.kaggle.com/wessam611)
