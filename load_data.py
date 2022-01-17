# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 18:53:20 2021

@author: Maysam
"""
import numpy as np
import scipy.io as sio
import random
import math
from spektral.data import Graph
from scipy.sparse import csr_matrix
from spektral.layers.ops import sp_matrix_to_sp_tensor

def read_data(dataset_name,path,modalities, split_range,dtype,ToSpTensor):
    
    x,a,y,N,modalities,N_modalities=load_matlab_data(dataset_name,path,modalities)

    randomlist = random.sample(range(0,N), N)
    
    # Public Planetoid splits. This is the default
    train_split=math.ceil(N*split_range[0])
    validation_split=math.ceil(N*split_range[1])
          
    # Public Planetoid splits. This is the default
    idx_tr = randomlist[0:train_split]
    idx_va = randomlist[train_split:train_split+validation_split]
    idx_te = randomlist[train_split+validation_split:N]
    idx_te_sort = np.sort(idx_te)
    
    mask_tr = _idx_to_mask(idx_tr,N)
    mask_va = _idx_to_mask(idx_va, N)
    mask_te = _idx_to_mask(idx_te, N)
    
    a_sparse=[csr_matrix(a[s]) for s in range(N_modalities)]
    
    G0=Graph(x=x[0].astype(dtype),a=a_sparse[0].astype(dtype),y=y.astype(dtype),)
    
    if ToSpTensor and G0.a is not None:
        G0.a = sp_matrix_to_sp_tensor(G0.a)
    
    return G0,mask_tr, mask_va, mask_te
    
def read_data_multiview(dataset_name,path,modalities, split_range,dtype,ToSpTensor):
    
    x,a,y,N,modalities,N_modalities=load_matlab_data(dataset_name,path,modalities)
    


    randomlist = random.sample(range(0,N), N)
    
    # Public Planetoid splits. This is the default
    train_split=math.ceil(N*split_range[0])
    validation_split=math.ceil(N*split_range[1])
          
    # Public Planetoid splits. This is the default
    idx_tr = randomlist[0:train_split]
    idx_va = randomlist[train_split:train_split+validation_split]
    idx_te = randomlist[train_split+validation_split:N]
    idx_te_sort = np.sort(idx_te)
    
    mask_tr = _idx_to_mask(idx_tr,N)
    mask_va = _idx_to_mask(idx_va, N)
    mask_te = _idx_to_mask(idx_te, N)
    
    a_sparse=[csr_matrix(a[s]) for s in range(N_modalities)]
       
    G=[Graph(x=x[s].astype(dtype),a=a_sparse[s].astype(dtype),y=y.astype(dtype),) for s in range(N_modalities)]
       
    for s in range(N_modalities):
        if ToSpTensor and G[s].a is not None:
            G[s].a = sp_matrix_to_sp_tensor(G[s].a)
    
    return G,mask_tr, mask_va, mask_te
    


def read_data_multimodal(dataset_name,path,modalities, split_range,dtype,ToSpTensor):
    
    x,a,y_temp,N,modalities,N_modalities=load_matlab_data(dataset_name,path,modalities)
    
    # create two y vectros for two modalities 
    y=[]
    y.append(y_temp)
    y.append(y_temp)
    
    #  rand order of modality 2
    idx_n=random.sample(range(0,N), N)
    x[1]=x[1][idx_n][:]
    a[1]=a[1][idx_n,:]  # 1) change order of rows
    a[1]=a[1][:,idx_n]  # 2) change order of columns
    y[1]=y_temp[idx_n]
    #
    
    #  rand order of modality 1
    idx_n=random.sample(range(0,N), N)
    x[0]=x[0][idx_n][:]
    a[0]=a[0][idx_n,:]  # 1) change order of rows
    a[0]=a[0][:,idx_n]  # 2) change order of columns
    y[0]=y_temp[idx_n]
    #
    
    
    # random list 
    randomlist = random.sample(range(0,N), N)
    
    # Public Planetoid splits. This is the default
    train_split=math.ceil(N*split_range[0])
    validation_split=math.ceil(N*split_range[1])
          
    # Public Planetoid splits. This is the default
    idx_tr = randomlist[0:train_split]
    idx_va = randomlist[train_split:train_split+validation_split]
    idx_te = randomlist[train_split+validation_split:N]
    idx_te_sort = np.sort(idx_te)
    
    mask_tr = _idx_to_mask(idx_tr,N)
    mask_va = _idx_to_mask(idx_va, N)
    mask_te = _idx_to_mask(idx_te, N)
    
    a_sparse=[csr_matrix(a[s]) for s in range(N_modalities)]
       
    G=[Graph(x=x[s].astype(dtype),a=a_sparse[s].astype(dtype),y=y[s].astype(dtype),) for s in range(N_modalities)]
       
    for s in range(N_modalities):
        if ToSpTensor and G[s].a is not None:
            G[s].a = sp_matrix_to_sp_tensor(G[s].a)
    
    return G,mask_tr, mask_va, mask_te
    




def _idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_matlab_data(dataset_name,path,modalities):
    mat_path = "{}\{}".format(path,dataset_name)
    
    dataList=sio.loadmat(mat_path)
    
    x = []
    a = []
        
    x = [dataList['x{}'.format(i)] for i in modalities]
    
    a = [dataList['a{}'.format(i)] for i in modalities]
    
    y=dataList['Y']

    N=int(dataList['N'])
    
    data=[x,a,y,N,modalities,len(modalities)]
    
    return data
    



def load_data(dataset_name,n_modality):
    """
    Loads input data from data directory

    All objects above must be saved using python pickle module.

    :param dataset_name: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """


    x = []
    a = []
    path="dataset"
    
    file_path_x = [_file_path(path, dataset_name, "x",  s) for s in range(1,n_modality+1)]
    x = [np.loadtxt(p,delimiter=',') for p in file_path_x]
    
    file_path_a = [_file_path(path, dataset_name, "a",  s) for s in range(1,n_modality+1)]
    a = [np.loadtxt(p,delimiter=',') for p in file_path_a]
    
    
    y=np.loadtxt("{}\{}\y.txt".format(path,dataset_name,),delimiter=',')
    
def load_Train_Test_data(dataset_name,n_modality):
    """
    Loads input data from data directory

    All objects above must be saved using python pickle module.

    :param dataset_name: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """


    xTrain = []
    xTest = []
    path="dataset"
    
    file_path_Train = [_file_path_train_test(path, "Train", dataset_name,  s) for s in range(1,n_modality+1)]
    xTrain = [np.loadtxt(p,delimiter=',') for p in file_path_Train]
    
    file_path_Test = [_file_path_train_test(path, "Test", dataset_name,  s) for s in range(1,n_modality+1)]
    xTest = [np.loadtxt(p,delimiter=',') for p in file_path_Test]
    
    
    yTrain=np.loadtxt("{}\{}\Train_y.txt".format(path,dataset_name,),delimiter=',')
    
    yTest=np.loadtxt("{}\{}\Test_y.txt".format(path,dataset_name,),delimiter=',')
    
def _file_path_train_test(path, dataset_type, dataset_name, s):
    return "{}\{}\{}{}_x.txt".format(path,dataset_name,dataset_type, s)

def _file_path(path, dataset_name,dataset_type, s):
    return "{}\{}\{}{}.txt".format(path,dataset_name,dataset_type, s)

