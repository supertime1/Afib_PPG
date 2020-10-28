import glob
import wfdb
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn import preprocessing
import scipy
from scipy import signal
import tensorflow as tf


def data_mining(file_path):
    """

    :param file_path: local path of ECG data and label files
    :return: records and corresponding labels for records that are >30s
    """
    data_path = os.path.join(file_path, "*.hea")
    label_path = os.path.join(file_path, "REFERENCE.csv")

    df = pd.read_csv(label_path, sep=',', header=None)
    raw_signals, raw_labels = [], []

    for name in glob.glob(data_path):
        position = name.index('.hea')
        name = name[0:position]
        record = wfdb.rdheader(name)

        # only needs data with more than 30s records (300Hz sampling frequency)
        if record.sig_len < 9000:
            continue
        else:
            temp_name = name[-6:]  # remove the file_path and keep only the filename
            temp_label = df[df[0] == temp_name][1].to_numpy()

            if temp_label == '~':
                continue
            elif temp_label == 'N':
                temp_label = '0'
            elif temp_label == 'A':
                temp_label = '1'
            elif temp_label == 'O':
                temp_label = '2'

            record = wfdb.rdrecord(name)
            raw_signals.append(record.p_signal)
            raw_labels.append(temp_label)

    return raw_signals, raw_labels


def generate_seg_data(raw_signals, raw_labels, seg_len):
    """

    :param raw_signals: output signals from data_mining()
    :param raw_labels: output labels from data_mining()
    :param seg_len: length of signal segments
    :return: segmented ECG signals with corresponding labels
    """
    signals, labels = [], []
    n = 0
    for signal in raw_signals:
        for i in range(int(len(signal) / seg_len)):
            seg = signal[(seg_len * i):(seg_len * (i + 1))]
            label = raw_labels[n]
            signals.append(seg)
            labels.append(label)
        n += 1

    signals = np.asarray(list(map(lambda x: np.reshape(x, 9000), signals)))
    labels = np.asarray(list(map(lambda x: np.reshape(x, 1), labels)))

    return signals, labels


def preprocessing(signals, labels, timedistributed=False):
    """

    :param signals: output segmented signals from generate_seg_data
    :param labels: corresponding labels of segmented signals
    :return: rescaled, resampled and reshaped dataset to feed into NN model
    """
    signals = [sklearn.preprocessing.robust_scale(i) for i in signals]
    signals = [scipy.signal.resample(i, 3750) for i in signals]
    signals = [np.expand_dims(i, axis=1) for i in signals]
    if timedistributed:
        signals = [np.reshape(i, (3, int(3750/3), 1)) for i in signals]
    labels = tf.keras.utils.to_categorical(labels, num_classes=3)

    signals = np.array(signals)
    labels = np.array(labels)
    return signals, labels


def split_shuffle_dataset(signals, labels, train_ratio, seed=10):
    """

    :param signals: output segmented signals from preprocessing
    :param labels: corresponding labels of segmented signals
    :param train_ratio: training data ratio
    :param seed: random seed for shuffling
    :return: train and test dataset and label
    """
    m = len(list(signals))
    train_size = int(train_ratio * m)

    # shuffle dataset
    np.random.seed(seed=seed)
    np.random.shuffle(signals)
    train_dataset = signals[:train_size]
    test_dataset = signals[train_size:]

    np.random.seed(seed=seed)
    np.random.shuffle(labels)
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    return train_dataset, train_labels, test_dataset, test_labels


def count_labels(labels):
    nsr = sum(1 for i in labels if np.argmax(i) == 0)
    af = sum(1 for i in labels if np.argmax(i) == 1)
    others = sum(1 for i in labels if np.argmax(i) == 2)
    print('There are {} NSR labels'.format(nsr))
    print('There are {} AF labels'.format(af))
    print('There are {} Other Arrhythmia labels'.format(others))

    return nsr, af, others

def class_weights(labels):
    nsr, af, others = count_labels(labels)
    unit = 1/(1/nsr + 1/af + 1/others)
    nsr_ratio = float(unit / nsr)
    af_ratio = float(unit / af)
    others_ratio = float(unit / others)
    CLASS_WEIGTHS = {0: nsr_ratio,
                     1: af_ratio,
                     2: others_ratio}
    return CLASS_WEIGTHS