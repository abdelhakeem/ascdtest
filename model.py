from tensorflow import keras
import librosa
import numpy as np


MODEL_PATH = 'model.h5'
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
CMDS = ['zoom out',
        'direction',
        'zoom in',
        'ok',
        'eight',
        'previous',
        'six',
        'stop',
        'right',
        'zero',
        'two',
        'nine',
        'undo',
        'seven',
        'down',
        'four',
        'left',
        'digit',
        'send',
        'receive',
        'up',
        'move',
        'backward',
        'forward',
        'cancel',
        'next',
        'disable',
        'open',
        'five',
        'record',
        'enter',
        'yes',
        'one',
        'no',
        'three',
        'enable',
        'rotate',
        'start',
        'options',
        'close',
        'silence']


def load_model():
    return keras.models.load_model(MODEL_PATH)


def predictions(model, signal, rate=16000):
    mfcc = librosa.feature.mfcc(y=signal[:rate], sr=rate,
                                n_mfcc=NUM_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    predictions = model.predict(mfcc)
    predictions = [(CMDS[i], prob) for i, prob in enumerate(predictions[0])]

    return sorted(predictions, key=lambda p: p[1], reverse=True)


def predict(model, signal, rate=16000):
    mfcc = librosa.feature.mfcc(y=signal[:rate], sr=rate,
                                n_mfcc=NUM_MFCC,
                                n_fft=N_FFT,
                                hop_length=HOP_LENGTH)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    predictions = model.predict(mfcc)
    predicted_index = np.argmax(predictions)
    predicted_keyword = CMDS[predicted_index]

    return predicted_keyword
