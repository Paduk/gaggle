import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

from specaugment import spec_augment
import numpy as np

# ref http://www.kwangsiklee.com/tag/model-fit_generator/
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, datas, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, valid=False):
        'Initialization'
        #print("Init datagen")
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.tdata = datas
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.valid = valid
        self.on_epoch_end()

    def __len__(self):
        #print("__len__")
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.tdata) / self.batch_size))

    def __getitem__(self, index):
        #print("__getitem__")
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]                        
        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]        
        # Generate data
        X, y = self.__data_generation(indexes)        
        return X, y

    def on_epoch_end(self):
        #print("on_epoch_end")
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.tdata))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, index):
        #print("__data_gen")
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data        
        for i, ID in enumerate(index):
            if self.valid:                                
                X[i] = self.tdata[ID]
            else:
                #X[i] = self.tdata[ID]
                X[i] = spec_augment(self.tdata[ID].reshape(1,40,101,1))
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)