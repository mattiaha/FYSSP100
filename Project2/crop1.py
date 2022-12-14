import gc
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
i = 0
files = glob.glob(r"C:/Users/matti/CTA/Data/1dc/models/DL_pics/small/*.png")
for my_file in files:
    image = Image.open(my_file).convert('RGB')
    width, height = image.size
    left = 190
    top = 230
    right = 1110
    bottom = 1150
    im1 = image.crop((left, top, right, bottom))
    im1.save("cropped"+str(i)+".png")
    i += 1
    

files = glob.glob(r"C:/Users/matti/CTA/Data/1dc/models/DL_pics/small_dm/*.png")
for my_file in files:
    image = Image.open(my_file).convert('RGB')
    width, height = image.size
    left = 190
    top = 230
    right = 1110
    bottom = 1150
    im1 = image.crop((left, top, right, bottom))
    im1.save("cropped"+str(i)+".png")
    i += 1

labels = np.zeros(12000,dtype = int)
for j in range(6000,12000):
    labels[j] = 1
np.save("labels.npy",labels)