#!/usr/bin/python
import sys
sys.path.append('../src')

####################
# Variables

import numpy as np

factor=0.5;
dim1=5;
dim2=7;
nch=2;

a=np.linspace(1,dim1*dim2*nch,dim1*dim2*nch);

input_data  = a.reshape((1,dim1,dim2,nch));

####################
# Creating the model

import tensorflow as tf
from SectorPooling.Layer2D import Sector4Pooling2D

input_shape=(dim1, dim2,nch);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=factor,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

####################
# Applying the model

output_data = model.predict(input_data);

print('input_data:\n',input_data);
print('input_data.shape:\n',input_data.shape);
print('output_data:\n',output_data);
print('output_data.shape:\n',output_data.shape);


