# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:59:45 2018

@author: PR
"""

import numpy as np
 
import tensorflow as tf
 
import random as rn
 
# The below is necessary in Python 3.2.3 onwards to
 
# have reproducible behavior for certain hash-based operations.
 
 
 
import os
 
os.environ['PYTHONHASHSEED'] = '0'
# The below is necessary for starting Numpy generated random numbers
 
# in a well-defined initial state.
np.random.seed(42)
# The below is necessary for starting core Python generated random numbers
 
# in a well-defined state.
rn.seed(42)
# Force TensorFlow to use single thread.
 
# Multiple threads are a potential source of
 
# non-reproducible results.
 
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
 
K.set_learning_phase(1)
# The below tf.set_random_seed() will make random number generation
 
# in the TensorFlow backend have a well-defined initial state.
 
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
 
K.set_session(sess)
# Convolutional Neural Network
# Part 1 - Building the CNN
# Importing the Keras libraries and packages
 
from keras import initializers
 
from keras.models import Sequential
 
from keras.layers import Conv2D
 
from keras.layers import MaxPooling2D
 
from keras.layers import Flatten
 
from keras.layers import Dense
 
from keras.layers import Dropout
 
from keras.layers import BatchNormalization
 
from keras.optimizers import adam
 
from keras.callbacks import ModelCheckpoint
 
import tensorflow as tf
# GPU and/or CPU identification
 
from tensorflow.python.client import device_lib
 
print(device_lib.list_local_devices())
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
classifier.add(Conv2D(64, (4, 4), input_shape = (128, 128, 3), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
 
# Step 2 - Pooling (first Batch normalization)
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (4, 4)))
 
# Adding a second convolutional layer
 
classifier.add(Conv2D(64, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
 
classifier.add(BatchNormalization())
 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a third convolutional layer
 
classifier.add(Conv2D(32, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
 
classifier.add(BatchNormalization())
 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Adding a fourth convolutional layer
 
classifier.add(Conv2D(32, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
 
classifier.add(BatchNormalization())
 
classifier.add(MaxPooling2D(pool_size = (2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection 1
 
classifier.add(Dense(units = 4096, activation = 'relu'))
 
classifier.add(BatchNormalization())
 
classifier.add(Dropout(rate=0.3, seed=42))
 
# Step 6 - Full connection 2
 
classifier.add(Dense(units = 1024, activation = 'relu'))
 
classifier.add(BatchNormalization())
 
classifier.add(Dense(units = 1, activation = 'sigmoid'))
 
# Compiling the CNN
 
optimizer_adam = adam(lr = 0.0005, decay = 0.0001)
 
classifier.compile(optimizer = optimizer_adam, loss = 'binary_crossentropy', metrics = ['accuracy'])
 
# Part 2 - Fitting the CNN to the images
 
batch_size = 32
 
stepPerEpoch = round(8000/batch_size)
 
validationSteps = round(2000/batch_size)
with tf.device('/gpu:0'):
 
    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,
 
                                       shear_range = 0.2,
 
                                       zoom_range = 0.2,
 
                                       horizontal_flip = True,
 
                                       rotation_range = 40,
 
                                       width_shift_range = 0.2,
 
                                       height_shift_range = 0.2,
 
                                       fill_mode = 'nearest',
 
                                       zca_epsilon = 0.15
 
                                       )
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('dataset/training_set',
 
                                 target_size = (128, 128),
 
                                 batch_size = batch_size,
 
                                 class_mode = 'binary',
 
                                 seed= None)
    test_set = test_datagen.flow_from_directory('dataset/test_set',
 
                                    target_size = (128, 128),
 
                                    batch_size = batch_size,
 
                                    class_mode = 'binary',
 
                                    seed=None)
    filepath ='checkpoint42_loss_acc-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
 
    checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode='auto')
    classifier.fit_generator(training_set,
 
                                 steps_per_epoch = stepPerEpoch,
 
                                 epochs = 250,
 
                                 validation_data = test_set,
 
                                 validation_steps = validationSteps,
 
                                 callbacks= [checkpoint],
 
                                 shuffle=True,
 
                                 workers=1,
 
                                 max_queue_size=1)
# freeze layers
 
for l in classifier.layers:
 
     l.trainable=False
# Save model
 
classifier.save('full_model42.h5')
 
model_yaml= classifier.to_yaml()
 
with open ('model42.yaml','w') as yaml_file:
 
    yaml_file.write(model_yaml)
 
classifier.save_weights('model42.h5')
 
print ("model and weights saved")