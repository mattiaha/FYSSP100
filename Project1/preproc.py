# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:27:20 2022

@author: matti
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:37:43 2022

@author: matti
"""



import numpy as np
import csv
import os

with open('C:/Users/matti/TF/training.csv', 'r') as f:
    data = list(csv.reader(f,delimiter = ','))


data = np.array(data)
header = data[0,:].copy()
#Delete 
data = np.delete(data,0,0)
num_rows, num_col = data.shape


#Convert from string to integer.
for i in range(num_rows):
    if (data[i,num_col-1] == 'b'):
        data[i,num_col-1] = 0
    else:
        data[i,num_col-1] = 1
        
# Remove columns we don't need.
data = np.delete(data,num_col-2,1)
data = np.delete(data,0,1)

data = data.astype(np.float64)


col_max = data.max(axis=0)

num_rows, num_col = data.shape

proc_data = np.copy(data)

#Randomize the rows
np.random.shuffle(proc_data)



#Set the -999 values to above max value so they don't interfere with finding minimum values for each column
for j in range(num_col-1):
    for k in range(num_rows):
        if (proc_data[k,j] == -999):
            proc_data[k,j] = col_max[j]+10

col_min = proc_data.min(axis=0)

#Subtract the minimum value for each column respectively. 
for j in range(num_col-1):
    for k in range(num_rows):
        proc_data[k,j] -= col_min[j]
        

#Entries that were -999 are now found as the new max values and are set to 0.
col_max = proc_data.max(axis=0)
for j in range(num_col-1):
    for k in range(num_rows):
        if (proc_data[k,j]==col_max[j]):
            proc_data[k,j] = 0

col_min2 = proc_data.min(axis=0)

#Normalize each column respectively.
col_max2 = proc_data.max(axis=0)
col_max2.astype(float)
for j in range(num_col-1):
    for k in range(num_rows):
        proc_data[k,j] = proc_data[k,j]/col_max2[j]
        
        

n = 10
data_set = np.vsplit(proc_data,n)

#Create folders to divide up training and validation data
for i in range(n):
    file_data = np.copy(data_set)
    dir_name = "folder"+str(i)
    v_name = "val.npy"
    t_name = "val_t.npy"
    try: 
        os.mkdir(dir_name) 
    except OSError as error: 
        print(error)
    directory = './'+dir_name+'/'
    val = file_data[i]
    file_data = np.delete(file_data,i,axis=0)
    val_target = val[:,num_col-1]
    val = np.delete(val,num_col-1,1)
    
    np.save(os.path.join(dir_name,v_name), val)
    np.save(os.path.join(dir_name,t_name), val_target)
    
    train_name  = "train.npy"
    target_name = "target.npy"
    reshaped_data = np.reshape(file_data,(int(num_rows - num_rows/10),num_col))
    
    target = reshaped_data[:,num_col-1]
    train  = np.delete(reshaped_data,num_col-1,1)
    
    np.save(os.path.join(dir_name,train_name), train)
    np.save(os.path.join(dir_name,target_name), target)

        
 
targets = proc_data[:,num_col-1]
proc_data = np.delete(proc_data,num_col-1,1)

np.save("../TF/Features_T.npy", proc_data)
np.save("../TF/Targets_T.npy", targets)
np.save("../TF/header.npy", header)

with open('C:/Users/matti/TF/test.csv', 'r') as f:
    Tdata = list(csv.reader(f,delimiter = ','))
    
Tdata = np.array(Tdata)
Tdata = np.delete(Tdata,0,0)
Tdata = np.delete(Tdata,0,1)
Tnum_rows, Tnum_col = Tdata.shape
Tdata = Tdata.astype(np.float64)


proc_data_T = np.copy(Tdata)
Tcol_max = Tdata.max(axis=0)

Tnum_rows, Tnum_col = Tdata.shape

#Set the -999 values to above max value so they don't interfere with finding minimum values for each column
for j in range(Tnum_col):
    for k in range(Tnum_rows):
        if (proc_data_T[k,j] == -999):
            proc_data_T[k,j] = Tcol_max[j]+10

Tcol_min = proc_data_T.min(axis=0)

#Subtract the minimum value for each column respectively. 
for j in range(Tnum_col):
    for k in range(Tnum_rows):
        proc_data_T[k,j] -= Tcol_min[j]
        

#Entries that were -999 are now found as the new max values and are set to 0.
Tcol_max = proc_data_T.max(axis=0)
for j in range(Tnum_col):
    for k in range(Tnum_rows):
        if (proc_data_T[k,j]==Tcol_max[j]):
            proc_data_T[k,j] = 0

Tcol_min2 = proc_data_T.min(axis=0)

#Normalize each column respectively.
Tcol_max2 = proc_data_T.max(axis=0)
Tcol_max2.astype(float)
for j in range(Tnum_col):
    for k in range(Tnum_rows):
        proc_data_T[k,j] = proc_data_T[k,j]/Tcol_max2[j]

np.save("../TF/Features_Test.npy", proc_data_T)

    
with open('C:/Users/matti/TF/submission.csv', 'r') as f:
    sub = list(csv.reader(f,delimiter = ','))  
    
test_targets = np.array(sub)
test_targets = test_targets[:,2]
test_targets = np.delete(test_targets,0)
num_rows = test_targets.size


#Convert from string to integer.
for i in range(num_rows):
    if (test_targets[i] == 'b'):
        test_targets[i] = 0
    else:
        test_targets[i] = 1
        
np.save("../TF/Targets_Test.npy", test_targets)

