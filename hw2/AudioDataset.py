import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchaudio_augmentations import (
#     RandomResizedCrop,
#     RandomApply,
#     PolarityInversion,
#     Noise,
#     Gain,
#     HighLowPass,
#     Delay,
#     PitchShift,
#     Reverb,
#     Compose,
# )

class AudioDataset(Dataset):
    def __init__(self, paths, input_length, binary, id_to_path, split, is_augmentation=False):
        self.paths = paths
        self.input_length = input_length
        self.binary = binary
        self.id_to_path = id_to_path
        self.split = split
        self.is_augmentation = is_augmentation
        if is_augmentation:
            self._get_augmentations()

    # def _get_augmentations(self):
    #     transforms = [
    #         RandomResizedCrop(n_samples=self.num_samples),
    #         RandomApply([PolarityInversion()], p=0.8),
    #         RandomApply([Noise(min_snr=0.3, max_snr=0.5)], p=0.3),
    #         RandomApply([Gain()], p=0.2),
    #         RandomApply([HighLowPass(sample_rate=16000)], p=0.8),
    #         RandomApply([Delay(sample_rate=16000)], p=0.5),
    #         RandomApply([PitchShift(n_samples=self.num_samples, sample_rate=16000)], p=0.4),
    #         RandomApply([Reverb(sample_rate=16000)], p=0.3),
    #     ]
    #     self.augmentation = Compose(transforms=transforms)

    def __getitem__(self, index):
        item = self.binary.iloc[index]
        waveform = self.item_to_waveform(item)
        return waveform.astype(np.float32), item.values.astype(np.float32)

    def item_to_waveform(self, item):
        id = item.name
        path = os.path.join(self.paths,
                            self.id_to_path[id].replace(".mp3", ".npy"))  # pre-extract waveform, for fast loader
        waveform = np.load(path)
        if self.split in ['TRAIN', 'VALID']:
            random_idx = np.random.randint(low=0, high=int(waveform.shape[0] - self.input_length))
            waveform = waveform[random_idx:random_idx + self.input_length]  # extract input
            audio = np.expand_dims(waveform, axis=0)  # 1 x samples
        elif self.split == 'TEST':
            sample_rate = 16000
            duration = 3
            input_length = sample_rate * duration
            chunk_number = waveform.shape[0] // self.input_length
            chunk = np.zeros((chunk_number, self.input_length))
            for idx in range(chunk.shape[0]):
                chunk[idx] = waveform[idx:idx + input_length]
            audio = chunk
        return audio

    def __len__(self):
        return len(self.binary)
