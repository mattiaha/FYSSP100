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
    left = 40
    top = 40
    right = 130
    bottom = 130
    im1 = image.crop((left, top, right, bottom))
    im1.save("croppedsmall/cropped"+str(i)+".png")
    i += 1
    

files = glob.glob(r"C:/Users/matti/CTA/Data/1dc/models/DL_pics/small_dm/*.png")
for my_file in files:
    image = Image.open(my_file).convert('RGB')
    width, height = image.size
    left = 40
    top = 40
    right = 130
    bottom = 130
    im1 = image.crop((left, top, right, bottom))
    im1.save("croppedsmall_dm/cropped"+str(i)+".png")
    i += 1

