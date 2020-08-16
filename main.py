import os
from os.path import isdir, join
from pathlib import Path
import pandas as pd
import random
# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import pandas as pd

#from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model

import csv
import h5py
import math

from pydub import AudioSegment
from specaugment import spec_augment, visualization_spectrogram
from custom_datagen import DataGenerator

from preprocessing import Preprocess
from model import Modeling
from postprocessing import Postprocess


def plot_history(history, filename):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Acc')
  plt.plot(hist['epoch'], hist['acc'],
           label='Train acc')
  plt.plot(hist['epoch'], hist['val_acc'],
           label = 'Val acc')  
  plt.legend()
  plt.savefig("acc_" + filename)
  plt.figure(figsize=(8,12))

  plt.subplot(2,1,1)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.plot(hist['epoch'], hist['loss'],
           label='Train loss')
  plt.plot(hist['epoch'], hist['val_loss'],
           label = 'Val loss')  
  plt.legend()
  plt.savefig("loss_" + filename)
  #plt.show()

def main():
	# Parameters
	params = {'dim': (40,101),
	          'batch_size': 256,
	          'n_classes': 31,
	          'n_channels': 1,
	          'shuffle': False}

	# PREPROCESSING
	pre_process = Preprocess()
	#pre_process.split_silence() # 1568 silence file is created.	

	#os.system("find dataset/train/audio/. | '.wav' > dataset/train/audio/kaggle_train.txt")
	pre_process.getwavelist() # 66289 is num of data
	#pre_process.makehdf5("dataset/train/preprocess.hdf5")
	pre_process.makehdf5_unknown("../dataset/train/preprocess_unknown.hdf5")

	
	t_data, t_label = pre_process.preprocessing("../dataset/train/preprocess_unknown.hdf5")
	
	percentage_t = int(0.9 * t_data.shape[0])
	tr_data, tr_label = t_data[:percentage_t], t_label[:percentage_t]
	v_data, v_label = t_data[percentage_t:], t_label[percentage_t:]

	training_generator = DataGenerator(t_data, t_label, **params)
	#training_generator = DataGenerator(tr_data, tr_label, **params)
	valid_generator = DataGenerator(v_data, v_label, **params, valid=True)
	
	# check numofdata per label
	for i in range(11):
		print(i, len(np.where(np.argmax(t_label, axis=-1) == i)[0]))

	# MODELING AND TRAINING
	model = Modeling()
	model.build_model()
	#model.build_model_VGG16()
	#model.load_model("dataset/train/hj_vgg16_spec.h5")
	history = model.train_model_augment(training_generator, valid_generator, epoch=10)
	#history = model.train_model(t_data, t_label, epoch=10)

	# save weights
	model.save_model("spec_unknown.h5")
	#model.load_model("dataset/train/hj_weights.h5")

	#result = model.predict_model(t_data[:512])
	#print(result)
	exit(0)
	# POSTPROCESSING
	post_process = Postprocess()
	post_process.makehdf5("../dataset/test/postprocess.hdf5") # only once is enough

	post_feature = post_process.loadhdf5("dataset/test/postprocess.hdf5")
	test_predict = np.zeros([post_process.numofdata], dtype=int)
	for idx, pf in enumerate(post_feature):
		pred_res = model.predict_model(pf)		
		test_predict[idx*10000:idx*10000+len(pred_res)] = pred_res
	
	post_process.postprocessing(test_predict)
if __name__ == "__main__":    
    main()


# normalization - 잘 안되는것 같어
# spec augmentation
# make model deeper - VGG16 되게하기 