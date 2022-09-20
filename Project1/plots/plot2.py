# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 11:30:24 2022

@author: matti
"""

import matplotlib.pyplot as plt
import csv
import numpy as np




with open('../TF/model3.csv', 'r') as f:
     data = list(csv.reader(f,delimiter = ';'))
data = np.array(data)


data = np.delete(data,0,0)
data = np.float64(data)
train_acc = data[:,1]
train_loss = data[:,2]
val_acc = data[:,3]
val_loss = data[:,4]

plt.figure(1)
x = np.array(range(len(val_acc)))
plt.plot(x,train_acc, label = 'Training')
plt.plot(x,val_acc, label = 'Validation')
plt.xlabel("Epochs")
plt.ylabel('Accuracy')
plt.title('Accuracy for model 3')
plt.legend()
plt.axis([0,300,0,1])

plt.show()

plt.figure(2)
x = np.array(range(len(val_acc)))
plt.plot(x,train_loss, label = 'Training')
plt.plot(x,val_loss, label = 'Validation')
plt.xlabel("Epochs")
plt.ylabel('Loss')
plt.title('Loss for model 3')
plt.legend()
plt.axis([0,300,0,1])
plt.show()

