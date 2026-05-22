#!/usr/bin/python
import sys
import SectorPooling.Model2D as mst

import os
import tensorflow as tf

FUNC_ACT=layer = tf.keras.layers.LeakyReLU();


def create_model_sectortree(file_of_weight='',input_shape=(224,224,3),nout=7,enable_summary=False, factor=0.5):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    
    block1 = tf.keras.Sequential([
        tf.keras.layers.Conv2D( 16, kernel_size=7, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=7, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=7, padding="same", activation=FUNC_ACT),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D( 16, kernel_size=5, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=5, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=5, padding="same", activation=FUNC_ACT),
    ])

    block2 = tf.keras.Sequential([
        tf.keras.layers.Conv2D( 16, kernel_size=5, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=5, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=5, padding="same", activation=FUNC_ACT),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=FUNC_ACT),
    ])

    block3 = tf.keras.Sequential([
        tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D( 16, kernel_size=1, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  8, kernel_size=1, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=1, padding="same", activation=FUNC_ACT),
    ])

    
    # modelo nuevo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Conv2D( 16, kernel_size=11, padding="same", activation=FUNC_ACT, input_shape=input_shape),
        tf.keras.layers.Conv2D(  4, kernel_size= 9, padding="same", activation=FUNC_ACT),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        #
        
        tf.keras.layers.Conv2D( 16, kernel_size=9, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=7, padding="same", activation=FUNC_ACT),
        tf.keras.layers.MaxPooling2D( pool_size=(2, 2)),
        
        tf.keras.layers.BatchNormalization(),
        
        
        mst.model_sectortree2d( input_shape=(56,56,4),
                                blocks=[block1,block2,block3],
                                factor=factor,
                                min_size=8,
                                name='SMT_',
                                to_file='salida.png'),

        
        #
        
        tf.keras.layers.Conv2D( 16, kernel_size=3, padding="same", activation=FUNC_ACT),
        tf.keras.layers.Conv2D(  4, kernel_size=3, padding="same", activation=FUNC_ACT),
        
        tf.keras.layers.Flatten(),
        
        #
        
        tf.keras.layers.Dense(nout*3+1,activation='tanh'),
        tf.keras.layers.Dense(nout    ,activation='softmax')

    ])
    
    if enable_summary:
        modelo.summary();
    
    if os.path.exists(file_of_weight):
        #if (len(file_of_weight)!=0):
        obj=modelo.load_weights(file_of_weight);
    
    return modelo
    
#create_model_sector4(file_of_weight='',input_shape=(224,224,3),nout=7);
