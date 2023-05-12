"""
Training the model with P-phase, noise(S-phase) dataset.
"""
import os
import sys

import tensorflow as tf

import keras
from keras import Input, layers, models
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
import keras_metrics as km

from core.io import get_dir_list
from core.generator import DataGenerator3_ex, DataGenerator3_ex2
from core.model import UNetPlusPlusPro
from core.losses import weighted_cross_entropy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pretty_errors
import ipdb as pdb

# epochs = 1   # for testing
epochs = 100    
datasetId = 'dataset_pdf_std_3ch'  # dataset used by training

# GPUS = 1
GPUS = 4
batchSize = 128  # 128, 256
beta = 5
loss = weighted_cross_entropy(beta)  # weight=20, 10, 5, 3, 2, 1

split_point = -50000  

weight_dir = "data/weights"
pkl_dir = "/home/data/ai/chuandian_%s" % datasetId    
trained_weight = "trained_weight_bs%d_adam_%s_ep%d_p&s_3ch_lossweight_beta%d.h5" %  (batchSize, datasetId, epochs, beta)    # chuandian

pkl_list = get_dir_list(pkl_dir)
pkl_list = [pkl for pkl in pkl_list if pkl.endswith('.pkl')]
np.random.seed(42)
np.random.shuffle(pkl_list)

training_generator = PredictGenerator3(pkl_list[:split_point], batch_size=batchSize, shuffle=True)
validation_generator = PredictGenerator3(pkl_list[split_point:], batch_size=batchSize)


fname = os.path.sep.join([weight_dir, "train_weights.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss",save_best_only=True, verbose=2)
lr_scheduler=LearningRateScheduler(_lr_schedule)
callbacks = [lr_scheduler]

if GPUS <= 1:
    print("[INFO] training with 1 GPU...")
    dim = (1, 3001, 3)
    model1 = UNetPlusPlusPro(*dim, num_class=2)  # The model for P-phase and noise 

    input = Input(shape=dim)
    output1 = model1(input)

    model = models.Model(input, output1) 

    opt = Adam(lr=_lr_schedule(0))    
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', km.binary_precision(), km.binary_recall(), km.binary_f1_score()])
    H = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                        epochs=epochs, use_multiprocessing=True)

    model.save_weights(os.path.join(weight_dir, trained_weight))
    print('Saving the model: %s' % trained_weight)    
    
else:
    print("[INFO] training with {} GPUs...".format(GPUS))
    with tf.device("/cpu:0"):
        dim = (1, 3001, 3)
        model1 = UNetPlusPlusPro(*dim, num_class=2)  # The model for P-phase and noise 

        input = Input(shape=dim)
        output1 = model1(input)

        model = models.Model(input, output1)   # for softmax
    parallel_model = multi_gpu_model(model, gpus=GPUS)   # multiple gpus

    opt = Adam(lr=_lr_schedule(0))    
    parallel_model.compile(optimizer=opt, loss=loss, metrics=['accuracy', km.binary_precision(), km.binary_recall(), km.binary_f1_score()] )

    H = parallel_model.fit_generator(generator=training_generator, validation_data=validation_generator,
                        epochs=epochs, use_multiprocessing=True, callbacks=callbacks)


    # os.makedirs(weight_dir, exist_ok=True)
    model.save_weights(os.path.join(weight_dir, trained_weight))           
    trained_weight_model = trained_weight[:trained_weight.rfind('.h5')] + '_model.h5'
    model.save(os.path.join(weight_dir, trained_weight_model))
    print('Saving the model: %s' % trained_weight)    
    
# Store the log for training process
df = pd.DataFrame(data=H.history)
df.to_csv(os.path.join(weight_dir, "trained_weight_log.csv"))










