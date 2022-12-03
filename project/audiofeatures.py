# GCT634 (2022) Project
# Nov-30-2022: initial version
# Vanessa Tan
import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from BeatNet.BeatNet import BeatNet
import matplotlib.pyplot as plt

#data_path = './aist_music/'
newdata_path = './aist_music_30/'
out_path = './beat_feat/'

def audioFeatures(file):
    # # Load File
    # y, sr = librosa.load(file, sr=16000)
    #
    # # STFT
    # STFT = np.abs(librosa.stft(y))
    #
    # # Dynamics/Loudness
    # rms = librosa.feature.rms(S=STFT) # [1, TIME]
    # zcr = librosa.feature.zero_crossing_rate(y) # [1, TIME]
    #
    # # Frequency Content
    # spec_centroid = librosa.feature.spectral_centroid(S=STFT, sr=sr) # [1, TIME]
    # spec_bw = librosa.feature.spectral_bandwidth(S=STFT, sr=sr) # [1, TIME]
    # spec_flat = librosa.feature.spectral_flatness(S=STFT) # [1, TIME]
    #
    # # MFCC + Deltas (n_mfcc = 20)
    # S = librosa.feature.melspectrogram(S=STFT, sr=sr)
    # S_dB = librosa.power_to_db(S, ref=np.max)
    # mfcc = librosa.feature.mfcc(S=S_dB, sr=sr) # [20, TIME]
    # mfcc_delta = librosa.feature.delta(mfcc) # [20, TIME]
    # mfcc_delta2 = librosa.feature.delta(mfcc, order=2) # [20, TIME]

    # Beat
    estimator = BeatNet(1, mode='online', inference_model='PF', plot=[], thread=False)
    output = estimator.process(file)

    # Plot Beat
    # plt.rcParams['figure.figsize'] = (20, 6)
    # audio, sr = librosa.load(file, sr=16000)
    # spec = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    # librosa.display.specshow(spec, y_axis='log', sr=sr, x_axis='time')
    # plt.title('Log-frequency power spectrogram of track')
    # plt.colorbar(format="%+2.f dB")
    # plt.vlines(output[:, 0][output[:, 1] == 2], 0, sr / 2, linestyles='dotted', color='w')
    # plt.vlines(output[:, 0][output[:, 1] == 1], 0, sr / 2, color='w')
    # plt.show()

    # Concatenate Features
    # features = [rms, zcr,
    #             spec_centroid, spec_bw, spec_flat,
    #             mfcc, mfcc_delta, mfcc_delta2]
    # features_array = np.concatenate(features, axis=0)
    return output.T

def extract_feat():
    for file in os.listdir(newdata_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(newdata_path, file)):
            # Trim Audio to 30s
            # y, sr = librosa.load(data_path + file, sr=16000,  duration=30)
            # sf.write(newdata_path + file , y, 16000, format='wav')
            features = audioFeatures(newdata_path + file)

            file_name = file.replace('.wav','.npy')
            save_file = out_path + file_name
            if not os.path.exists(os.path.dirname(save_file)):
                os.makedirs(os.path.dirname(save_file))
            np.save(save_file, features)
        print("Processing: " + file)

if __name__ == '__main__':
    # feat = np.load(acoustic_path + 'mBR0.npy')
    # print(feat.shape)

    #audioFeatures('aist_music_30/mWA1.wav')
    extract_feat()
