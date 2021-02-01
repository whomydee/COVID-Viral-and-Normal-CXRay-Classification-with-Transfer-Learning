#from google.colab import drive
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
import pickle
import os
#from progressbar import progressbar
import copy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time

image_size = 128

new_model = tf.keras.models.load_model('C:/Users/shad_/Desktop/2. Fall 2020/CSE 6211 (Deep Learning)/Submission/Project/Covid Detection with UNet/Models/basic_model')
#new_model.summary()

tmp = open("C:/Users/shad_/Desktop/2. Fall 2020/CSE 6211 (Deep Learning)/Submission/Project/Covid Detection with UNet/Dumps/testing_x_dump.pickle", "rb")
test_x = pickle.load(tmp)

test_x /= 255.0

print(test_x.shape)

test_x = np.asarray(test_x).reshape(test_x.shape[0], image_size, image_size, 1)

shads = new_model.predict(test_x)
covid = 0
viral = 0
normal = 0
for shad in shads:
    if((shad[0] >= shad[1]) and (shad[0] >= shad[2])):
        covid += 1
    elif((shad[1] >= shad[0]) and (shad[1] >= shad[2])):
        viral += 1
    else:
        normal += 1

print("COVID: " + str(covid))
print("VIRAL: " + str(viral))
print("NORMAL: " + str(normal))

