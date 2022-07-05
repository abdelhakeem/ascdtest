# Live Testing for Arabic Speech Command Recognition

Live testing app for [this](https://www.kaggle.com/code/abdelhakeemelgamal/arabic-kws-updated?scriptVersionId=99946247) Arabic keyword spotting model.

## Quickstart

```
$ virtualenv .env
$ source .env/bin/activate
$ pip install -r requirements.txt
$ python main.py
```

## Dataset

Original Dataset: https://github.com/abdulkaderghandoura/arabic-speech-commands-dataset

The dataset is a list of pairs (x, y) where x is the input speech signal, and y is the corresponding keyword. The final dataset consists of 12000 such pairs, comprising 40 keywords. Each audio file is one second in length sampled at 16 kHz.
There were 30 participants, each of them recorded 10 utterances for each keyword. Therefore, we have 300 audio files for each keyword in total (30 × 10 × 40 = 12000). Lastly, the total size of the dataset is ~384 MB.
The table below lists the 40 chosen keywords with their translations into Arabic and pronunciations in the International PhoneticAlphabet (IPA):

<p align="center">
  <img src="https://user-images.githubusercontent.com/20467669/177334899-6afd7c77-8dc8-42d7-b689-ca6282e073bf.png" />
</p>

As commonly done in machine learning settings, we split the dataset into three subsets: training, validation, and testing.

Considering the number of instances in our dataset, we decided to keep 80% of them as the training set, 10% as the validation set, and the remaining 10% are kept as a hold-out testing set.

In our split method, we guarantee that all recordings of a certain contributor are within the same subset. In this way, we avoid having signals with some similarities in both the training and validation/testing sets, as this may affect the validity of the results. Besides, it makes sure that the model will learn to generalize to new people outside of our dataset.

## Data Augmentation

We combined and used several data augmentation techniques over 10 rounds with a probability of 0.5 for each augmentation to make up for the low volume of data:

- Add gaussian noise to the samples
- Time stretch the signal without changing the pitch
- Shift the samples forwards or backwards
- Frequency masking
- Time masking

We also added 3000 silent segments with some Gaussian noise to the dataset to be able to detect silence.
