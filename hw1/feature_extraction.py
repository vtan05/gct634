# GCT634 (2018) HW1
# Mar-18-2018: initial version
# Juhan Nam
#
# Sept-20-2022: edited version
# Vanessa Tan

import sys
import os
import numpy as np
import librosa

data_path = './dataset/'
features_path = './features/'

def compute_features(y, sr):

    # STFT
    D = np.abs(librosa.stft(y))
    D_dB = librosa.amplitude_to_db(D, ref=np.max)

    # RMS
    rms = librosa.feature.rms(S=D)

    # Chroma
    chroma_stft = librosa.feature.chroma_stft(S=D, sr=sr)  # STFT
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)  # CQT
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)  # Chroma Energy Normalized (CENS)

    # Mel-Spectrogram (n_mels = 128)
    S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # MFCC + Deltas (n_mfcc = 13)
    mfcc13_128 = librosa.feature.mfcc(S=S_dB, n_mfcc=13)
    mfcc13_128_delta = librosa.feature.delta(mfcc13_128)
    mfcc13_128_delta2 = librosa.feature.delta(mfcc13_128, order=2)
    # MFCC + Deltas (n_mfcc = 26)
    mfcc26_128 = librosa.feature.mfcc(S=S_dB, n_mfcc=26)
    mfcc26_128_delta = librosa.feature.delta(mfcc26_128)
    mfcc26_128_delta2 = librosa.feature.delta(mfcc26_128, order=2)

    # Mel-Spectrogram (n_mels = 256)
    S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=256)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # MFCC + Deltas (n_mfcc = 13)
    mfcc13_256 = librosa.feature.mfcc(S=S_dB, n_mfcc=13)
    mfcc13_256_delta = librosa.feature.delta(mfcc13_256)
    mfcc13_256_delta2 = librosa.feature.delta(mfcc13_256, order=2)
    # MFCC + Deltas (n_mfcc = 26)
    mfcc26_256 = librosa.feature.mfcc(S=S_dB, n_mfcc=26)
    mfcc26_256_delta = librosa.feature.delta(mfcc26_256)
    mfcc26_256_delta2 = librosa.feature.delta(mfcc26_256, order=2)

    # Centroid and Bandwidth
    spec_centroid = librosa.feature.spectral_centroid(S=D)
    spec_bw = librosa.feature.spectral_bandwidth(S=D)

    # Flatness and Rolloff
    spec_flat = librosa.feature.spectral_flatness(S=D)
    rolloff = librosa.feature.spectral_rolloff(y=y, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(y=y, roll_percent=0.01)

    # Polynomial Features
    p0 = librosa.feature.poly_features(S=D, order=0)
    p1 = librosa.feature.poly_features(S=D, order=1)
    p2 = librosa.feature.poly_features(S=D, order=2)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Tonal Centroids
    harmonics = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonics, sr=sr)

    # Spectral Flux Onset Strength
    onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=[0, 32, 64, 96, 128])

    # Concatenate Features
    features = [rms, chroma_stft, chroma_cq, chroma_cens,
                mfcc13_128, mfcc13_128_delta, mfcc13_128_delta2,
                mfcc26_128, mfcc26_128_delta, mfcc26_128_delta2,
                mfcc13_256, mfcc13_256_delta, mfcc13_256_delta2,
                mfcc26_256, mfcc26_256_delta, mfcc26_256_delta2,
                spec_centroid, spec_bw, spec_flat, rolloff, rolloff_min,
                p0, p1, p2, zcr, tonnetz, onset_subbands]
    features_array = np.concatenate(features, axis=0)
    #print(features_array.shape)

    return features_array

def extract_features(dataset='train'):
    f = open(data_path + dataset + '_list.txt','r')

    i = 0
    for file_name in f:
        # progress check
        i = i + 1
        if not (i % 10):
            print(i)

        # load audio file
        file_name = file_name.rstrip('\n')
        file_path = data_path + file_name
        y, sr = librosa.load(file_path, sr=22050)

        # save features as a file
        file_name = file_name.replace('.wav','.npy')
        save_file = features_path + file_name

        # extract features
        features = compute_features(y, sr)

        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        np.save(save_file, features)

    f.close();

if __name__ == '__main__':
    extract_features(dataset='train')
    extract_features(dataset='valid')

