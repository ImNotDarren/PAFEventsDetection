import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Dense,
                                     Conv1D,
                                     InputLayer,
                                     MaxPooling1D,
                                     GlobalAveragePooling1D,
                                     Dropout,
                                     SeparableConv1D,
                                     BatchNormalization,
                                     Bidirectional,
                                     LSTM,
                                     Reshape)

from biosppy.signals.tools import filter_signal
from biosppy.signals import ecg

class CNN:
    def __init__(self):
        self.model = Sequential()
        self.build_model()
        self.callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath='models.{epoch:02d}-{val_loss:.2f}.h5',
                save_best_only=True
            )
        ]

    def load_weights(self, weight_path):
        # with open(model_path, 'r') as file:
        #     model_json = file.read()
        # self.model = model_from_json(model_json)
        self.model.load_weights(weight_path)


    # build cnn
    def build_model(self):
        self.model.add(InputLayer(input_shape=(1500, 2)))
        self.model.add(Conv1D(32, 15, activation='relu'))
        self.model.add(Conv1D(32, 15, activation='relu'))
        self.model.add(MaxPooling1D(2, padding='same'))
        self.model.add(Conv1D(64, 15, activation='relu'))
        self.model.add(Conv1D(64, 15, activation='relu'))
        self.model.add(MaxPooling1D(2, padding='same'))
        self.model.add(BatchNormalization(epsilon=0.01))
        self.model.add(Reshape((-1, 128)))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))  # af, non-af

    def get_train(self, train):
        Xs = []
        ys = []
        # get X and y
        # scan the first eight heartbeats to predict the eighth
        for i, t in enumerate(train):
            print(str(i + 1) + '/' + str(len(train)), end='\r')
            rpeaks = t['rpeaks']
            # from the eighth peak
            for p in range(9, len(rpeaks)):
                end = rpeaks[p]
                start = end - 1499
                if start < 0:
                    start = 0
                    end = 1499
                sig = t['sig_filtered'][start:end + 1]
                # classification on p-1 location
                if t['class_true'] == 1:
                    y = 1
                elif t['class_true'] == 2:
                    # if p-1 is between an af start and end point, y = 1
                    for j in range(len(t['af_start_scripts'])):
                        if p - 1 in range(t['af_start_scripts'][j], t['af_end_scripts'][j] + 1):
                            y = 1
                            break
                else:
                    y = 0
                Xs.append(sig)
                ys.append(y)

        Xs = np.array(Xs)
        ys = np.array(ys)

        # with open('../data/Xs', 'wb') as file:
        #     pickle.dump(self.Xs, file)
        #
        # with open('../data/ys', 'wb') as file:
        #     pickle.dump(self.ys, file)

        return Xs, ys

    def save_model(self):
        model_json = self.model.to_json()
        with open('cnn_model.json', 'w') as file:
            file.write(model_json)

    def fit(self, train, batch_size=20, epochs=10, verbose=1, validation_split=0.2):
        Xs, ys = self.get_train(train)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        print(self.model.summary())

        self.save_model()

        self.model.fit(Xs,
                       to_categorical(ys, 2),
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_split=validation_split,
                       callbacks=self.callbacks)

    def score(self, test):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.load_weights('cnn_model.h5')
        # get test X and y
        Xs, ys = self.get_train(test)
        evaluation = self.model.evaluate(Xs, to_categorical(ys, 2), batch_size=20, verbose=1, return_dict=True)
        print(evaluation)
