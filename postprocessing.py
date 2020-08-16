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

class Postprocess:

	def __init__(self):
		self.pathlist = "kaggle.txt"
		self.base_path = "dataset/test/audio/"
		self.submission_file = "hj_submission.csv"
		self.frame_length = 0.025
		self.frame_stride = 0.010
		self.sample_rate = 16000
		self.input_nfft = int(round(self.sample_rate * self.frame_length))
		self.input_stride = int(round(self.sample_rate * self.frame_stride))		
		self.labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

		IdLookupTable = pd.read_csv("dataset/sample_submission.csv", na_values = "?", comment='\t', sep=",", skipinitialspace=True)
		self.Fname = IdLookupTable["fname"]
		self.numofdata = len(self.Fname)		


	def makehdf5(self, f_towrite):		
		f = open(self.base_path + self.pathlist, 'r')
		filelist = f.readlines()
		test_data = np.zeros([self.numofdata, 40, 101], dtype=float)

		for idx, fl in enumerate(filelist):
		    fl = fl[:-1]

		    samples, sample_rate = librosa.load(self.base_path + fl, sr=self.sample_rate)
		    S = librosa.feature.melspectrogram(y=samples, n_mels=40, n_fft=self.input_nfft, hop_length=self.input_stride)
		    if S.shape[1] != 101:
		        S = librosa.util.fix_length(S, 101, axis=1)  # zero-paddings

		    test_data[idx] = S
		    if idx % 2000 == 0:
		        print(idx)

		with h5py.File(f_towrite, 'w') as hf:
		    dy_str = h5py.special_dtype(vlen=str)
		    fn_data = hf.create_dataset('feature', [self.numofdata, 40, 101], dtype=float)
		    fn_data[:self.numofdata] = test_data[:self.numofdata]

		f.close()

	def loadhdf5(self, f_toread):
		with h5py.File(f_toread, 'r') as hf:
			for i in range(math.ceil(self.numofdata / 10000)):
				if (i+1)*10000 > self.numofdata:
					feature = hf.get('feature')[i*10000 : self.numofdata]
					yield feature.reshape(self.numofdata - i*10000, 40, 101, 1)
				else:
		  			feature = hf.get('feature')[i*10000 : (i+1)*10000]
		  			yield feature.reshape(10000, 40, 101, 1)


	def postprocessing(self, predict_res):
		f = open(self.submission_file, 'w', newline='')		
		wr = csv.writer(f)
		wr.writerow(['fname','label'])

		for i in range(self.numofdata):
			if predict_res[i] > 10:
				pred_label = "unknown"
			else:
				pred_label = self.labels[predict_res[i]]
			wr.writerow([self.Fname[i], pred_label]) 

		f.close()
