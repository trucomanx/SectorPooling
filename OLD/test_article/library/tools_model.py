#!/usr/bin/python

import sys

import numpy as np

import model_sectortree  as mt
import model_sectortree2 as mt2
import model_sectortree3 as mt3
import model_sector4 as ms
import model_max     as mm  

def create_model(   file_of_weight='',
                    model_type='model_sector4',
                    input_shape=(224,224,3),
                    nout=7,
                    enable_summary=False):
    '''
    Retorna un modelo para la clasificación.
    Adicionalmente, si el archivo `file_of_weight` existe los pesos son cargados.
    
    :param file_of_weight: Archivo donde se encuentran los pesos.
    :type file_of_weight: str
    :return: Retorna un modelo de red neuronal
    :rtype: tensorflow.python.keras.engine.sequential.Sequential
    '''
    
    # una capa cualquiera en tf2
    if  model_type=='model_sector4':
        model = ms.create_model_sector4(file_of_weight=file_of_weight,
                                        input_shape=input_shape,
                                        nout=nout,
                                        enable_summary=enable_summary);
        print("Loaded model_sector4");
    
    elif  model_type=='model_sectortree':
        model = mt.create_model_sectortree( file_of_weight=file_of_weight,
                                            input_shape=input_shape,
                                            nout=nout,
                                            enable_summary=enable_summary);
        print("Loaded model_sectortree");
    
    elif  model_type=='model_sectortree2':
        model = mt2.create_model_sectortree2(file_of_weight=file_of_weight,
                                            input_shape=input_shape,
                                            nout=nout,
                                            factor=0.53,
                                            enable_summary=enable_summary);
        print("Loaded model_sectortree2");
    
    elif  model_type=='model_sectortree3':
        model = mt3.create_model_sectortree3(file_of_weight=file_of_weight,
                                            input_shape=input_shape,
                                            nout=nout,
                                            factor=0.53,
                                            enable_summary=enable_summary);
        print("Loaded model_sectortree3");
    
    elif model_type=='model_max':
        model = mm.create_model_max(file_of_weight=file_of_weight,
                                    input_shape=input_shape,
                                    nout=nout,
                                    enable_summary=enable_summary);
        print("Loaded model_max");
    
    else:
        model = ms.create_model_sector4(file_of_weight=file_of_weight,
                                        input_shape=input_shape,
                                        nout=nout,
                                        enable_summary=enable_summary);
        print("Loaded model_sector4");
    
    target_size=(input_shape[0],input_shape[1]);
    
    return model,target_size;


def save_model_stat_kfold(mydict, fpath,sep=','):
    outdict={};
    with open(fpath, 'w') as csv_file:  
        myvalues=[];
        mykeys=[];
        for key, value in mydict.items():
            mykeys.append(key);
            myvalues.append(value);
            outdict[key+'_MEAN']=np.mean(value);
        
        N=len(mykeys);
        L=len(myvalues[0]);
        
        for n in range(N):
            csv_file.write(mykeys[n]);
            if n<N-1 :
                csv_file.write(sep);
            else:
                csv_file.write('\n');
        
        for l in range(L):
            for n in range(N):
                csv_file.write(str(myvalues[n][l]));
                if n<N-1 :
                    csv_file.write(sep);
                else:
                    csv_file.write('\n');

    return outdict;

from keras.utils.layer_utils import count_params
def save_model_parameters(model, fpath):
    '''
    Salva en un archivo la estadistica de la cantidoda de parametros de un modelo
    
    :param model: Modelos a analizar
    :type model: str
    :param fpath: Archivo donde se salvaran los datos.
    :type fpath: str
    '''
    trainable_count = count_params(model.trainable_weights)
    
    fid = open(fpath, 'w')
    print('parameters_total={}'.format(model.count_params()),';', file = fid);
    print('parameters_trainable={}'.format(trainable_count),';', file = fid);
    fid.close()


import matplotlib.pyplot as plt

def save_model_history(hist, fpath,show=True, labels=['accuracy','loss']):
    ''''This function saves the history returned by model.fit to a tab-
    delimited file, where model is a keras model'''

    acc      = hist.history[labels[0]];
    val_acc  = hist.history['val_'+labels[0]];
    loss     = hist.history[labels[1]];
    val_loss = hist.history['val_'+labels[1]];

    EPOCAS=len(acc);
    
    rango_epocas=range(EPOCAS);

    plt.figure(figsize=(16,8))
    #
    plt.subplot(1,2,1)
    plt.plot(rango_epocas,    acc,label=labels[0]+' training')
    plt.plot(rango_epocas,val_acc,label=labels[0]+' validation')
    plt.legend(loc='lower right')
    #plt.title('Analysis accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    #
    plt.subplot(1,2,2)
    plt.plot(rango_epocas,    loss,label=labels[1]+' training')
    plt.plot(rango_epocas,val_loss,label=labels[1]+' validation')
    plt.legend(loc='lower right')
    #plt.title('Analysis loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    #
    plt.savefig(fpath+'.plot.png')
    if show:
        plt.show()
    
    print('max_val_acc', np.max(val_acc))
    
    ###########
    
    # Open file
    fid = open(fpath, 'w')
    print('accuracy,val_accuracy,loss,val_loss', file = fid)

    try:
        # Iterate through
        for i in rango_epocas:
            print('{},{},{},{}'.format(acc[i],val_acc[i],loss[i],val_loss[i]),file = fid)
    except KeyError:
        print('<no history found>', file = fid)

    # Close file
    fid.close()
    
    return acc, val_acc
