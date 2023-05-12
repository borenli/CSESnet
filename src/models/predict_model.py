import os, sys
import warnings
import datetime

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

from keras.optimizers import Adam, SGD

from core.io import get_dir_list
from core.pick import write_pdf_to_dataset3
from core.generator import PredictGenerator3
from core.model import UNetPlusPlusPro

import ipdb as pdb
import pretty_errors


batchSize = 100  
length=30   # the length of the waveform in 30s, default 30 
lengthPts = length*100 + 1   # 100-sampling rate. defaults 3001

weight_file = "/home/data/CSESnet/models/trained_weight_bs128_adam(chuandian)_pdf_std_lossweight_beta5_pro.h5"   # model chuandian (P phase)
pkl_dir = "/home/data/pkl/test_random"      # p三分向数据

pkl_output_dir = pkl_dir + "_predict"
pkl_list = get_dir_list(pkl_dir)
pkl_list = [pkl for pkl in pkl_list if pkl.endswith('.pkl')]

predict_generator = PredictGenerator3(pkl_list, batch_size=batchSize, dim=(1, lengthPts))

# Select a type model
dim = (1, lengthPts, 3)
model = UNetPlusPlusPro_2Net(dim=dim)          # for sigmoid (P-phase and Noise)


model.load_weights(weight_file)
starttime = datetime.datetime.now()   # compute the running time

predict = model.predict_generator(generator=predict_generator,
                                  use_multiprocessing=True, verbose=True)

endtime = datetime.datetime.now()     # compute the running time
timtRunning  = (endtime - starttime).seconds

height = 0.3    
el
if DEBUG:
    write_pdf_to_dataset(predict, pkl_list, pkl_output_dir, pick_phase='S', remove_dir=True, height=height, distance=100)

else:
    write_pdf_to_dataset3(predict, pkl_list, pkl_output_dir, remove_dir=True, height=height, distance=100)  # for sigmoid (P-phase and Noise(noise, S-phase))

print('seisnn3 running time: %f(s)' % timtRunning)