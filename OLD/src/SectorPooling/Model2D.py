#!/usr/bin/python

import sys
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from SectorPooling.Layer2D import SectorNPooling2D

def change_layer_list_labels(layer_list,prefix):
    for n in range(len(layer_list)):
        layer_list[n]._name = prefix + layer_list[n]._name;
        try: 
            change_layer_list_labels(layer_list[n].layers,prefix);
        except:
            pass;

def change_model_labels(model,prefix):
    model._name = prefix+model._name;
    change_layer_list_labels(model.layers,prefix);


def sectortree2d_recursive_func(entrada,blocks:list,factor:float,name:str):
    SectorList=[None,None,None,None];
    out=[];
    
    for n in range(4):
        name_n=name+str(n);
        
        tmp_flow = SectorNPooling2D(factor=factor,sector=n)(entrada);
        
        model_block=tf.keras.models.clone_model(blocks[0]);
        #model_block._name=name_n;
        
        change_model_labels(model_block,prefix=name_n);
        
        tmp_flow = model_block(tmp_flow);
        
        if len(blocks)>1:
            new_blocks=blocks[1:];
            SectorList[n]=sectortree2d_recursive_func(tmp_flow,new_blocks,factor,name_n);
        else:
            SectorList[n]=[tmp_flow];
        
        out=out+SectorList[n];
    
    return out;


def model_sectortree2d(input_shape,blocks,factor=0.618,name=None,to_file=None):
    if not isinstance(name, str):
        name=str(time.time_ns())+'_';
    
    entrada = tf.keras.layers.Input(shape=input_shape);
    
    out=sectortree2d_recursive_func(entrada,blocks,factor,name);
    
    # If output Dense:  out[0].shape (None, 8) 
    # If output Conv2D: out[0].shape (None, 19, 19, 4)
    salida=K.concatenate(out, axis=(len(out[0].shape)-1));
    
    model = tf.keras.Model(inputs=entrada, outputs=salida)
    
    if isinstance(to_file, str):
        tf.keras.utils.plot_model(model, to_file=to_file,dpi=200,show_shapes=True)
    
    return model






