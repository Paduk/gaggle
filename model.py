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


class Modeling:

	def __init(self):
		pass

	def build_model_VGG16(self):  
		self.model = keras.Sequential([
			layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=[40,101,1]),
			layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2'),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  name='block1_pool'),		

			layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1'),
			layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2'),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  name='block2_pool'),

			layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1'),
			layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2'),
			layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3'),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  name='block3_pool'),

			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1'),
			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2'),
			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3'),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  name='block4_pool'),

			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1'),
			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2'),
			layers.Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3'),
			layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  name='block5_pool'),

			layers.Flatten(name='flatten'),
			layers.Dense(4096, activation='relu', name='fc1'),
			layers.Dense(4096, activation='relu', name='fc2'),
			layers.Dropout(0.5),	
			layers.Dense(31, activation='softmax', name='predictions'),
		])
  		
		optimizer = tf.keras.optimizers.Adam()
		self.model.compile(loss='categorical_crossentropy',
                	optimizer=optimizer,
                	metrics=['acc'])

		print(self.model.summary())

	def build_model(self):  
		self.model = keras.Sequential([
			layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=[40,101,1]),
		    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),		    
		    layers.Dropout(0.2),
		    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
		    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
		    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
		    layers.Dropout(0.3),
		    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
		    layers.Flatten(),
		    layers.Dense(256, activation='relu'),
		    layers.Dropout(0.3),
		    layers.Dense(31, activation='softmax')
		])
  		
		optimizer = tf.keras.optimizers.Adam()
		self.model.compile(loss='categorical_crossentropy',
                	optimizer=optimizer,
                	metrics=['acc'])

		print(self.model.summary())


	def load_model(self, weight):
		self.model.load_weights(weight)


	def save_model(self, filename):
		self.model.save_weights(filename)


	def train_model_augment(self, train_gen, val_gen, epoch=5, v_split=0.1):		
		history = self.model.fit_generator(generator=train_gen, validation_data=val_gen, use_multiprocessing=False,
                    workers=6, epochs=epoch, verbose=1)		
		return history


	def train_model(self, train_data, train_label, bs=256, epoch=5, v_split=0.1):
		cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5", monitor='val_loss',
	                            verbose=1, save_best_only=True)
		history = self.model.fit(train_data, train_label, shuffle=True, batch_size=bs,
								epochs=epoch, validation_split=v_split, verbose=1, callbacks=[cb_checkpoint])		
		
		return history


	def predict_model(self, test_data, bs=512):
		predict_result = self.model.predict(test_data, batch_size=bs, verbose=1)
		pred_res = np.argmax(predict_result, axis=-1)		
		return pred_res

