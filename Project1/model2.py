# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 12:50:19 2022

@author: matti
"""

# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.callbacks import CSVLogger


csv_logger = CSVLogger('model2.csv', append=True, separator=';')

training =  np.load("../TF/folder0/train.npy")
val = np.load("../TF/folder0/val.npy")
training_t =  np.load("../TF/folder0/target.npy")
val_t = np.load("../TF/folder0/val_t.npy")

model = tf.keras.Sequential([
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(50, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')
    
    






        ])

model.compile(optimizer= tf.keras.optimizers.Adam(),
           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
           metrics=['accuracy'])

history = model.fit(training, training_t, epochs=300, 
                validation_data=(val, val_t), callbacks=[csv_logger])

