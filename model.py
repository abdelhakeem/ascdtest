from tensorflow import keras
import librosa
import numpy as np


MODEL_PATH = 'model.h5'
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
CMDS = ['backward',
        'cancel',
        'close',
        'digit',
        'direction',
        'disable',
        'down',
        'eight',
        'enable',
        'enter',
        'five',
        'forward',
        'four',
        'left',
        'move',
        'next',
        'nine',
        'no',
        'ok',
        'one',
        'open',
        'options',
        'previous',
        'receive',
        'record',
        'right',
        'rotate',
        'send',
        'seven',
        'six',
        'start',
        'stop',
        'three',
        'two',
        'undo',
        'up',
        'yes',
        'zero',
        'zoom in',
        'zoom out',
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
