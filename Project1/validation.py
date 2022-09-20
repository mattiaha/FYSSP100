# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from keras.callbacks import CSVLogger

for i in range(10):
    a = str(i)
    csv_logger = CSVLogger('log'  + a + '.csv', append=True, separator=';')

    training =  np.load("../TF/folder" +a+ "/train.npy")
    val = np.load("../TF/folder" + a + "/val.npy")
    training_t =  np.load("../TF/folder" +a+ "/target.npy")
    val_t = np.load("../TF/folder"+a+"/val_t.npy")

    model = tf.keras.Sequential([
         tf.keras.layers.Dense(10, activation='relu'),
         tf.keras.layers.Dense(10, activation='relu'),
         tf.keras.layers.Dense(10, activation='relu'),
         tf.keras.layers.Dense(1, activation='sigmoid')
        
        






            ])

    model.compile(optimizer= tf.keras.optimizers.Adam(),
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
               metrics=['accuracy'])

    history = model.fit(training, training_t, epochs=30, 
                    validation_data=(val, val_t), callbacks=[csv_logger])


