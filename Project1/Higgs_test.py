
"""
Created on Mon Aug  8 12:02:33 2022

@author: matti
"""
import numpy as np


training =  np.load("../TF/Features_T.npy")
test = np.load("../TF/Features_Test.npy")
training_t =  np.load("../TF/Targets_T.npy")
test_t = np.load("../TF/Targets_Test.npy")

test_t = test_t.astype(np.float64)


import tensorflow as tf
from tensorflow.keras import layers

from keras.callbacks import CSVLogger

csv_logger = CSVLogger('model1.csv', append=True, separator=';')
model = tf.keras.Sequential([
     tf.keras.layers.Dense(10, activation='relu'),
     tf.keras.layers.Dense(10, activation='relu'),
     tf.keras.layers.Dense(10, activation='relu'),
     tf.keras.layers.Dense(1, activation='sigmoid')
     





        ])

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0005),
           loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
           metrics=['accuracy'])

model.fit(training, training_t, epochs=100,
                validation_data=(test, test_t), callbacks=[csv_logger])

model.save('../TF/my_model')

                                    