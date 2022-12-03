import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from biosppy.signals.tools import filter_signal
from biosppy.signals import ecg
import json


def get_train(train):
    new_train = []
    for i in range(len(train)):
        print(str(i + 1) + '/' + str(len(train)), end='\r')
        tmp = dict()
        tmp['record_name'] = train[i]['record_name']
        tmp['fs'] = train[i]['fs']
        tmp['sig_filtered'] = np.transpose([filter_signal(train[i]['sig'][:, 0],
                                                          ftype='FIR',
                                                          band='bandpass',
                                                          order=50,
                                                          frequency=[0.5, 45],
                                                          sampling_rate=train[i]['fs'])[0],
                                            filter_signal(train[i]['sig'][:, 1],
                                                          ftype='FIR',
                                                          band='bandpass',
                                                          order=50,
                                                          frequency=[0.5, 45],
                                                          sampling_rate=train[i]['fs'])[0]
                                            ])
        tmp['rpeaks'] = ecg.christov_segmenter(tmp['sig_filtered'][:, 0], train[i]['fs'])[0].tolist()
        tmp['sig_filtered'] = tmp['sig_filtered'].tolist()
        tmp['beat_loc'] = train[i]['beat_loc'].tolist()
        tmp['af_start_scripts'] = train[i]['af_start_scripts'].tolist()
        tmp['af_end_scripts'] = train[i]['af_end_scripts'].tolist()
        tmp['class_true'] = train[i]['class_true']

        new_train.append(tmp)

    return new_train

if __name__ == '__main__':
    TRAINING_DATA_PATH = '../data/train.pkl'
    with open(TRAINING_DATA_PATH, 'rb') as file:
        train = pickle.load(file)

    new_train = get_train(train)

    with open('../data/train.json', 'w') as file:
        json.dump(new_train, file)

    print('Preprocessing succeed!')
