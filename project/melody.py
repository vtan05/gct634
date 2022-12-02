import torch
import argparse
import numpy as np
from MSnet.MelodyExtraction import MeExt
import os
import matplotlib.pyplot as plt
import matplotlib

data_path = './aist_music_30/'
model_path = './MSnet/pretrain_model/MSnet_' + str('melody')
output_dir = './output/'

def extractMelody(filepath):
    filename = filepath.split('/')[-1].split('.')[0]
    estimate = MeExt(filepath, model_type='melody', model_path=model_path, GPU=False, mode='std')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Save the result in '+ output_dir +'/' + filename + '.txt')
    np.savetxt(output_dir + '/' + filename + '.txt', estimate)

if __name__ == '__main__':
    #for file in os.listdir(data_path):
        # check if current path is a file
    #    if os.path.isfile(os.path.join(data_path, file)):
    #        extractMelody(data_path + file)

    estimate = MeExt('mWA1.wav', model_type='melody', model_path=model_path, GPU=False, mode='std')
    print(estimate.shape)
    print(estimate)

    plt.plot(estimate)
    plt.show()
    #imwrite('mWA1', (255 * image).astype(np.uint8))
