#!/usr/bin/python

import sys
sys.path.append('src')


import tensorflow as tf
from SectorPooling import Sector4Pooling2D

input_shape=(512, 512,3);

model = tf.keras.Sequential([
    Sector4Pooling2D(factor=0.5,input_shape=input_shape)
])

model.compile(loss='crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

