#!/usr/bin/python
import sys
sys.path.append('../src')

####################
# Variables
from PIL import Image
import numpy as np

im=Image.open('example.png');

image      = np.asarray(im);
print(image.shape)
input_data = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]));

####################
# Creating the model

factor=0.618;
sector=3;

import tensorflow as tf
from SectorPooling.Layer2D import SectorNPooling2D

input_shape=image.shape;

model = tf.keras.Sequential([
    SectorNPooling2D(factor=factor,sector=sector,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

####################
# Applying the model

output_data = model.predict(input_data);

output_data = output_data.astype('uint8');

####################
# Plotting the model

import matplotlib.pyplot as plt

print('input_data.shape:\n',input_data.shape);
fig1 = plt.figure()
plt.imshow(input_data[0]);

print('output_data.shape:\n',output_data.shape);
fig2 = plt.figure()
plt.imshow(output_data[0]);

plt.show();

