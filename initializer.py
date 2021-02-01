import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import os
from progressbar import progressbar
import copy
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers
from tensorflow.keras import Input

location_dataset = "_assets/Partitioned COVID-19 Radiography Database-Kaggle/"
location_covid_images = location_dataset + "covid/"
location_viral_images = location_dataset + "viral/"
location_normal_images = location_dataset + "normal/"

image_size = 128