# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 10:14:58 2022

@author: matti
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
n = 10

train_acc = np.zeros(n)
train_loss = np.zeros(n)
val_acc = np.zeros(n)
val_loss = np.zeros(n)

acc = np.zeros(n)
loss = np.zeros(n)

for i in range(n):
    with open('../TF/log' + str(i) +'.csv', 'r') as f:
        data = list(csv.reader(f,delimiter = ';'))
    data = np.delete(data,0,0)
    data = np.array(data)
    
    data = np.float64(data)
    train_acc[i] = data[-1,1]
    train_loss[i] = data[-1,2]
    val_acc[i] = data[-1,3]
    val_loss[i] = data[-1,4]
    acc[i] = abs(train_acc[i]-val_acc[i])
    loss[i] = abs(train_loss[i]-val_loss[i])
    
    
plt.figure(1)
x = np.linspace(1,len(val_acc),len(val_acc))
plt.plot(x,train_acc, "o", label = 'training')
plt.plot(x,val_acc, "o", label = 'validation')
plt.xlabel("Validation set")
plt.ylabel('Accuracy')
plt.title('Accuracy for cross validation')
plt.axis([0,10,0,1])
plt.legend()
plt.show()
   
plt.figure(2)
plt.plot(x,train_loss, "o", label = 'training')
plt.plot(x,val_loss, "o", label = 'validation')
plt.xlabel("Validation set")
plt.ylabel('Loss')
plt.title('Loss for cross validation')
plt.axis([0,10,0,1])
plt.legend()
plt.show()

mean_acc = np.mean(acc)
mean_loss = np.mean(loss)

print(mean_acc)
print(mean_loss)