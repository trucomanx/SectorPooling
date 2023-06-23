#!/usr/bin/python

import sys
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from SectorPooling.Layer2D import SectorNPooling2D


def sectortree2d_recursive_func(entrada,blocks,factor,min_size,name):
    work=[None,None,None,None];
    out=[];
    
    for n in range(4):
        tmp = SectorNPooling2D(factor=factor,sector=n)(entrada);
        
        model_block=tf.keras.models.clone_model(blocks[0]);
        
        model_block._name=name+str(n);
        
        tmp = model_block(tmp);
        
        new_blocks=blocks;
        if len(blocks)>1:
            new_blocks=blocks[1:];
        
        if tmp.shape[1]*factor>=min_size and tmp.shape[2]*factor>=min_size:
            work[n]=sectortree2d_recursive_func(tmp,new_blocks,factor,min_size,name+str(n));
        else:
            work[n]=[tmp];
        
        out=out+work[n];
    
    return out;


def model_sectortree2d(input_shape,blocks,factor=0.618,min_size=9,name=None,to_file=None):
    if not isinstance(name, str):
        name=str(time.time_ns())+'_';
    
    entrada = tf.keras.layers.Input(shape=input_shape);
    
    out=sectortree2d_recursive_func(entrada,blocks,factor,min_size,name);
    
    salida=K.concatenate(out, axis=3);
    
    model = tf.keras.Model(inputs=entrada, outputs=salida)
    
    if isinstance(to_file, str):
        tf.keras.utils.plot_model(model, to_file=to_file,dpi=200,show_shapes=True)
    
    return model





