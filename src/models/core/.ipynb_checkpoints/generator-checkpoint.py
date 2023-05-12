import pickle
import numpy as np

from obspy import read

from keras.utils import Sequence

from tqdm import tqdm
from tqdm.contrib import tenumerate
import ipdb as pdb


class DataGenerator3(Sequence):
    """
    Data generator for the P/S picker model using 3 channels for sigmoid
    """
    def __init__(self, pkl_list, batch_size=32, dim=(1, 3001), channels=3, shuffle=False):
        self.dim = dim     # 无论对于单通道还是三通道波形来说, dim=(1, 3001),通道数由 channels指定
        self.channels = channels
        self.batch_size = batch_size
        self.pkl_list = pkl_list
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.pkl_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.pkl_list) / self.batch_size))
   

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes]
        wavefile, probability_p, probability_s = self.__data_generation(temp_pkl_list)
        
        return wavefile, [probability_p, probability_s]

   
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

            
    def __data_generation(self, temp_pkl_list):
#         wavefile = np.empty((self.batch_size, *self.dim, self.channels))
#         probability_p = np.empty((self.batch_size, *self.dim, 1))  # The pdf of P-phase 
#         probability_s = np.empty((self.batch_size, *self.dim, 1))  # The pdf of s-phase 
        wavefile = np.zeros((self.batch_size, *self.dim, self.channels))
        probability_p = np.zeros((self.batch_size, *self.dim, 1))  # The pdf of P-phase 
        probability_s = np.zeros((self.batch_size, *self.dim, 1))  # The pdf of s-phase 
        for i, ID in enumerate(temp_pkl_list):
            with open(ID, 'rb') as fp:
                stream = pickle.load(fp)
#             pdb.set_trace()
            wavefile[i, :,:, 0] = stream[0].data.reshape(self.dim)    # EW
            wavefile[i, :,:, 1] = stream[1].data.reshape(self.dim)    # NS
            wavefile[i, :,:, 2] = stream[2].data.reshape(self.dim)    # 垂直向数据
            probability_p[i, :,:, 0] = stream.pdf[0].reshape(self.dim)    # corresponding to P-phase  
            probability_s[i, :,:, 0] = stream.pdf[1].reshape(self.dim)    # corresponding to S-phase

        return wavefile, probability_p, probability_s    

    
    
class PredictGenerator3(DataGenerator3):
    """
    Generator for prediction model, using 3 channels
    """     
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_pkl_list = [self.pkl_list[k] for k in indexes] 
        wavefile = self.__data_generation(temp_pkl_list)

        return wavefile


    def __data_generation(self, temp_pkl_list):
#         wavefile = np.empty((self.batch_size, *self.dim, self.channels))
        wavefile = np.zeros((self.batch_size, *self.dim, self.channels))        
        for i, ID in enumerate(temp_pkl_list):
            with open(ID, 'rb') as fp:
                stream = pickle.load(fp)                
            wavefile[i, :,:, 0] = stream[0].data.reshape(self.dim)    # EW
            wavefile[i, :,:, 1] = stream[1].data.reshape(self.dim)    # NS
            wavefile[i, :,:, 2] = stream[2].data.reshape(self.dim)    # 垂直向数据       

        return wavefile
    

   