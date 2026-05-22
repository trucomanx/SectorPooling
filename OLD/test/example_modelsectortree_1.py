#!/usr/bin/python
import sys
sys.path.append('../src')


import tensorflow as tf
from SectorPooling.Model2D import model_sectortree2d
from tensorflow.keras.layers import LeakyReLU

block1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 16, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=9, padding="same", activation=LeakyReLU()),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D( 16, kernel_size=7, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=7, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=7, padding="same", activation=LeakyReLU()),
])

block2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 16, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=5, padding="same", activation=LeakyReLU()),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=LeakyReLU()),
])

block3 = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=LeakyReLU()),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D( 16, kernel_size=1, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  8, kernel_size=1, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Conv2D(  4, kernel_size=1, padding="same", activation=LeakyReLU()),
])

factor=0.55;
min_size=5;
input_shape=(128,128,3);

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D( 3, kernel_size=9, padding="same", activation=LeakyReLU(), input_shape=input_shape),
    model_sectortree2d( input_shape=(128,128,3),
                        blocks=[block1,block2,block3],
                        factor=factor,
                        min_size=min_size,
                        name='LST',
                        to_file='layer_tree.png'),
    tf.keras.layers.Conv2D( 64, kernel_size=1, padding="same", activation=LeakyReLU()),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
])


model.summary()







