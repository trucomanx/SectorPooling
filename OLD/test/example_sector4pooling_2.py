#!/usr/bin/python
import sys
sys.path.append('../src')

####################
# Variables
from PIL import Image
import numpy as np

image      = np.asarray(Image.open('Eye-Testing-Chart.png'));
input_data = image.reshape((1,image.shape[0],image.shape[1],1));

####################
# Creating the model

factor=0.618;

import tensorflow as tf
from SectorPooling.Layer2D import Sector4Pooling2D

input_shape=(image.shape[0], image.shape[1],1);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=factor,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

####################
# Applying the model

output_data = model.predict(input_data);

####################
# Plotting the model

import matplotlib.pyplot as plt

print('input_data.shape:\n',input_data.shape);
fig1 = plt.figure()
plt.imshow(input_data[0,:,:,0],cmap='gray');

print('output_data.shape:\n',output_data.shape);
fig2 = plt.figure()
for n in range(4):
    ax = fig2.add_subplot(2, 2, n+1)
    plt.imshow(output_data[0,:,:,n],cmap='gray');
    plt.title('n:'+str(n))
plt.show();

