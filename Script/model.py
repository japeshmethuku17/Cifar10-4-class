# -*- coding: utf-8 -*-

# %tensorflow_version 2.x

# import tensorflow and tensorflow.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# import ImageDataGenerator and the related functions required for processing images
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# import all the layers required for building the CNN model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# import optimizers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# import statements for building and loading the model
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.models import model_from_json

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import he_normal, he_uniform, glorot_uniform

# import statements for one-hot encoding, model plotting
from tensorflow.keras.utils import to_categorical, plot_model

# import statements for loading ResNet50 from keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# import scikit-learn for metrics and the reports
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


import scipy.misc
import os 
import numpy as np
import pandas as pd
import zipfile
import csv
import cv2

# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# import drive to access the data from GDrive
#from google.colab import drive

# import seaborn
import seaborn as sns
from time import time

print("tensorflow version:",tf.__version__)

#Loading images from the dataset in Google Colaboratory

# Mounting the drive to the Colab Notebook for accessing the data
drive.mount('/content/drive/', force_remount=True)

# Unzipping the folder conatining the images and data
image_data = zipfile.ZipFile("/content/drive/My Drive/Cifar4_SortedImages.zip", 'r')
image_data.extractall("/tmp")
image_data.close()

# Specifying the location of images after extraction
train_dir = '/tmp/Cifar4_SortedImages/train/'
val_dir = '/tmp/Cifar4_SortedImages/validation/'
test_dir = '/tmp/Cifar4_SortedImages/test/'


# Displaying the number of samples available for training, validation and testing.
train_samples = sum(len(files) for _, _, files in os.walk(train_dir))
val_samples = sum(len(files) for _, _, files in os.walk(val_dir))
test_samples = sum(len(files) for _, _, files in os.walk(test_dir))

print("Number of images in training set:",train_samples)
print("Number of images in validation set:",val_samples)
print("Number of images in test set:",test_samples)

#Global Parameters
img_w, img_h = 32, 32
num_channels = 3
epochs = 400
batch_size = 16
randseed = 1

# Model Creation

def create_model():
  model = models.Sequential()
  model.add(Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_uniform', padding='same', input_shape=(img_w, img_h, num_channels)))
  model.add(BatchNormalization())
  model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.4))
  model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
  model.add(BatchNormalization())
  model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same'))
  model.add(BatchNormalization())
  model.add(MaxPooling2D((2,2)))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
  model.add(BatchNormalization())
  model.add(Dropout(0.6))
  model.add(Dense(4, activation='softmax'))
  return model

model = create_model()

#Visualizing the summary of CNN model
model.summary()

#Plotting the model to obatin the figure of CNN architecture
plot_model(model, to_file='model_figure.png', show_shapes=True, show_layer_names=True)

# Model Compilation

# Using Stochastic Gradient Descent as optimization algorithm
opt = SGD(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

# Specifying the callbacks to save the best model.
callbacks_list= [keras.callbacks.ModelCheckpoint('/content/drive/My Drive/Cifar_Model.hdf5', 
                                                 monitor='val_accuracy', 
                                                 verbose=1, 
                                                 save_best_only=True)]

# Augmenting the training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    height_shift_range=0.1,
    width_shift_range = 0.1,
    horizontal_flip=True)

#Validation and test images should not be augmented
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#Loading the augmented training images
training_data = train_datagen.flow_from_directory(train_dir,
                                                  class_mode='categorical',
                                                  target_size=(img_w, img_h),
                                                  seed=randseed,
                                                  batch_size=batch_size)
#Loading the images for validation
validation_data = val_datagen.flow_from_directory(val_dir,
                                                  class_mode='categorical',
                                                  target_size=(img_w, img_h),
                                                  seed=randseed,
                                                  batch_size=batch_size)
#Loading the images for testing
test_data = test_datagen.flow_from_directory(test_dir,
                                                  class_mode='categorical',
                                                  target_size=(img_w, img_h),
                                                  seed=randseed,
                                                  batch_size=batch_size)

# Train the CNN Model

# Fit the compiled model on the training data and validate with validation data
import math
t0 = time()

history= model.fit(training_data,
                   steps_per_epoch = math.ceil(train_samples/batch_size),
                   epochs = epochs,
                   validation_data = validation_data,
                   validation_steps = math.ceil(val_samples/batch_size),
                   callbacks= callbacks_list)

train_time = int(time() - t0)
print("Training time:"+train_time+"seconds")


# Visualize the execution results

# Visualize model's accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.minorticks_on()
plt.grid()
plt.figure()
# save image
plt.savefig('Classification Model Accuracy', dpi=250)
plt.show()

# Visualize model's loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.minorticks_on()
plt.grid()
plt.figure()
# save image
plt.savefig('Classification Model Loss', dpi=250)
plt.show()

# Model Evaluation

# Load the model for evaluation
cifar_model = load_model('./Model/Cifar_Model.hdf5')

#Model Validation
val_results = cifar_model.evaluate_generator(
    validation_data,
    steps=val_samples)

# Validation Accuracy is used for tuning the hyperparameters of the model
print('Validation accuracy:', (val_results[1]*100.0))
print('Validation loss:', (val_results[0]*100.0))

#Model Evaluation
test_results = cifar_model.evaluate_generator(
    test_data,
    steps=test_samples)

# Test accuracy is used to evaluate the model's performance
print('Test accuracy:', (test_results[1]*100.0))
print('Test loss:', (test_results[0]*100.0))
