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


class Preprocess:	

	def __init__(self):		
		self.pathlist = "kaggle_train.txt"
		self.base_path = "dataset/train/audio/"
		self.frame_length = 0.025
		self.frame_stride = 0.010
		self.sample_rate = 16000
		self.input_nfft = int(round(self.sample_rate * self.frame_length))
		self.input_stride = int(round(self.sample_rate * self.frame_stride))
		self.labels = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9, 'silence':10}	
		self.labels_unknown = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, 'on':6, 'off':7, 'stop':8, 'go':9, 'silence':10, 'unknown':11}	

	def getwavelist(self):
		f = open(self.base_path + self.pathlist,'r')
		self.filelist = f.readlines()
		random.shuffle(self.filelist)

		self.numofdata = len(self.filelist)
		print(self.numofdata, "is num of data")


	def split_silence(self):
		basic_path = "../dataset/train/audio/_background_noise_/"
		dest_path = "../dataset/train/audio/silence/"

		sil_fl = os.listdir(basic_path)
		sil_fl.remove("README.md")

		numof_silfile = 0
		for fidx, fl in enumerate(sil_fl):    
			s, sr = librosa.load(basic_path + fl, sr=16000)
			filelen = int(len(s)/sr)

			audio = AudioSegment.from_wav(basic_path + fl)

			for sidx in range(0, (filelen-1)*1000, 250):
				newAudio = audio[sidx:sidx+1000]
				newAudio.export(dest_path + '%d_%d.wav' % (fidx, sidx) , format="wav")
				numof_silfile += 1

		print("%d silence file is created." % numof_silfile)
        
        
	def makehdf5_unknown(self, f_towrite):				
		train_data = np.zeros([self.numofdata, 40, 101], dtype=float)
		train_label = np.zeros([self.numofdata], dtype=int)
		
		total_cnt = 0
		unknown_cnt = 0
		for fl in self.filelist:
		    fl = fl[:-1]
		    lab = self.labels.get(fl.split("/")[1])    
		    
		    if str(lab) == "None":
		    	if unknown_cnt >= 3000:
		    		continue

		        train_label[total_cnt] = 11
		        unknown_cnt += 1
		    else:        
		        train_label[total_cnt] = lab
		        
		    samples, sample_rate = librosa.load(self.base_path + fl, sr=16000)
		    S = librosa.feature.melspectrogram(y=samples, n_mels=40, n_fft=self.input_nfft, hop_length=self.input_stride)
		    if S.shape[1] != 101:
		        S = librosa.util.fix_length(S, 101, axis=1) # zero-paddings
		        
		    train_data[total_cnt] = S
		    total_cnt += 1
		    if total_cnt%2000 == 0 :
		    	print(total_cnt)

		with h5py.File(f_towrite, 'w') as hf:
		    dy_str = h5py.special_dtype(vlen=str)
		    fn_data = hf.create_dataset('feature', [total_cnt, 40, 101], dtype=float)
		    fn_data[:total_cnt] = train_data[:total_cnt]

		    fn_label = hf.create_dataset('label', [total_cnt], dtype=float)
		    fn_label[:total_cnt] = train_label[:total_cnt]
                
                
	def makehdf5(self, f_towrite):				
		train_data = np.zeros([self.numofdata, 40, 101], dtype=float)
		train_label = np.zeros([self.numofdata], dtype=int)
		
		total_cnt = 0
		for fl in self.filelist:
		    fl = fl[:-1]
		    lab = self.labels.get(fl.split("/")[1])    
		    
		    if str(lab) == "None":		        
		        self.labels[fl.split("/")[1]] = len(self.labels)  
		        train_label[total_cnt] = str(self.labels.get(fl.split("/")[1]))
		    else:        
		        train_label[total_cnt] = lab
		        
		    samples, sample_rate = librosa.load(self.base_path + fl, sr=16000)
		    S = librosa.feature.melspectrogram(y=samples, n_mels=40, n_fft=self.input_nfft, hop_length=self.input_stride)
		    if S.shape[1] != 101:
		        S = librosa.util.fix_length(S, 101, axis=1) # zero-paddings
		        
		    train_data[total_cnt] = S
		    total_cnt += 1
		    if total_cnt%2000 == 0 :
		    	print(total_cnt)
		

		with h5py.File(f_towrite, 'w') as hf:
		    dy_str = h5py.special_dtype(vlen=str)
		    fn_data = hf.create_dataset('feature', [self.numofdata, 40, 101], dtype=float)
		    fn_data[:self.numofdata] = train_data[:self.numofdata]

		    fn_label = hf.create_dataset('label', [self.numofdata], dtype=float)
		    fn_label[:self.numofdata] = train_label[:self.numofdata]


	def preprocessing(self, f_toread):					
		with h5py.File(f_toread, 'r') as hf:
			self.numofdata = len(hf.get('feature'))
			train_data = np.zeros([self.numofdata, 40, 101], dtype=float)
			train_label = np.zeros([self.numofdata], dtype=int)
			for i in range(math.ceil(self.numofdata / 10000)):
				if (i+1)*10000 > self.numofdata:
					train_data[i*10000 : self.numofdata] = hf.get('feature')[i*10000 : self.numofdata]
					train_label[i*10000 : self.numofdata] = hf.get('label')[i*10000 : self.numofdata]
				else:
				  	train_data[i*10000 : (i+1)*10000] = hf.get('feature')[i*10000 : (i+1)*10000]				  	
				  	train_label[i*10000 : (i+1)*10000] = hf.get('label')[i*10000 : (i+1)*10000]
		
		t_data  = train_data.reshape(self.numofdata, 40, 101, 1)
		#t_label = keras.utils.to_categorical(train_label, 31)

		# Normalize
		#print(t_data.shape, t_label.shape) # for debugging
		"""
		mean_np = np.mean(t_data, axis=0)
		mean_std = np.std(t_data, axis=0)
		t_data = (t_data - mean_np) / mean_std # normalizing
		"""
		return t_data, train_label

