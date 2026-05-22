#!/usr/bin/python
import sys
import json
import os


def load_dataset(dataset_name="mcfer2023",cpu_name='cpulab'):
    path_base=os.path.dirname(os.path.abspath(__file__));
    
    f = open(os.path.join(path_base,dataset_name+'_'+cpu_name+'.json'));
    
    data = json.load(f)
    f.close()
        
    dataset_csv_train_file = data['dataset_csv_train_file'];
    dataset_csv_test_file  = data['dataset_csv_test_file'];
    dataset_train_base_dir = data['dataset_train_base_dir'];
    input_shape=tuple(data['input_shape']);
    nout=data['nout'];
    
    return dataset_csv_train_file, dataset_csv_test_file,dataset_train_base_dir,input_shape,nout;


#print(load_dataset(dataset_name="ck+48",cpu_name='cpufer'))
