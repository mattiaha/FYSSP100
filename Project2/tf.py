# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 15:21:58 2022

@author: matti
"""
import gc
import datetime
import tensorflow as tf
import pathlib
from tensorflow.keras import datasets, layers, models
import numpy as np
import tensorboard as tb
from tensorflow.keras.callbacks import CSVLogger
import os
import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.config.list_physical_devices())
img_height = 920
img_width = 920
batch_size = 4
path = 'C:/Users/matti/CTA/Data/1dc/models/DL_pics/DL_project/croppedimages/'
#class a in the directory is without dm and class b is with

#image_count = len(list(path.glob('*.png')))
#print(image_count)      #12000
training_ds = tf.keras.utils.image_dataset_from_directory(
  path,
  labels = 'inferred',
  validation_split=0.1,
  shuffle = True,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

valid_ds =  tf.keras.utils.image_dataset_from_directory(
  path,
  labels = 'inferred',
  validation_split=0.1,
  shuffle = True,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


ep = 20



METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

csv_logger = CSVLogger('dm_model1.csv', append=True, separator=';')
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height,img_width,3)))
model.add(layers.BatchNormalization(
    axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0005),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[METRICS],              )
model.summary()

history = model.fit(training_ds, epochs=ep, 
                    validation_data=valid_ds,
                    batch_size=batch_size,
                    callbacks=[csv_logger])
model.save('dm_model1')
plt.figure(1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
plt.figure(2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.show()
plt.figure(3)
plt.plot(history.history['auc'], label = 'auc')
plt.plot(history.history['val_auc'])
plt.xlabel('Epoch')
plt.ylabel('AUC')
#plt.legend(loc='lower right')
plt.show()
plt.figure(4)
plt.plot(history.history['prc'], label = 'prc')
plt.plot(history.history['val_prc'])
plt.xlabel('Epoch')
plt.ylabel('PRC')
#plt.legend(loc='lower right')
plt.show()







