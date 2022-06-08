import math
import numpy as np
import pandas as pd
#import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, RootMeanSquaredError, MeanSquaredError
#from keras.wrappers.scikit_learn import KerasRegressor
#import scikeras
#from scikeras.wrappers import KerasRegressor
#import sklearn
#from sklearn.preprocessing import (StandardScaler, Normalizer)
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.model_selection import GridSearchCV, KFold
#from sklearn.pipeline import Pipeline

def MakeBinary(yTr):
    output = []
    tophalf = len(yTr[0])//2
    for y in yTr:
        tmp = np.zeros((len(yTr[0]),))
        idx = np.argsort(y)[::-1]
        for i in range(tophalf):
            tmp[idx[i]] = 1
        output.append(np.copy(tmp))
    return np.array(output)

def dataloader(xfile, yfile, sec_code_file):
    xTr = pd.read_csv(xfile)
    yTr = pd.read_csv(yfile)

    xTr = xTr.filter(like='Return_Close').to_numpy()
    #xTr = xTr.to_numpy()[::,1:]
    yTr = yTr.to_numpy()

    yTr = MakeBinary(yTr)
    print(xTr.shape, yTr.shape)

    sec_code_list = pd.read_csv(sec_code_file, header = None, dtype=str).to_numpy()

    return xTr, yTr, sec_code_list

def FormTimeWindow(xTr, yTr, window):
    yTr = yTr[window-1:]
    xTr_tmp = []
    for i in range(window-1,len(xTr)):
        xTr_tmp.append(np.reshape(xTr[i-window+1:i+1],(window*len(xTr[i]),)))
    xTr_tmp = np.array(xTr_tmp)    

    xTe = xTr_tmp[1080:]
    yTe = yTr[1080:]
 
    xTr = xTr_tmp[:1080]
    yTr = yTr[:1080]

    return xTr, yTr, xTe, yTe

def MovingAverage(xTr):
    # p*[2/(day+1)] + a*[1-2/(day+1)]
    ema5  = np.zeros(len(xTr[0]))
    ema10 = np.zeros(len(xTr[0]))
    ema20 = np.zeros(len(xTr[0]))
    
    ema5lst, ema10lst, ema20lst = [], [], []
    for x in xTr:
        ema5  = x*(2.0/(5.0+1.0)) + ema5*(1.0-2.0/(5.0+1.0))
        ema10 = x*(2.0/(10.0+1.0)) + ema5*(1.0-2.0/(10.0+1.0))
        ema20 = x*(2.0/(20.0+1.0)) + ema5*(1.0-2.0/(20.0+1.0))
        
        ema5lst.append(np.copy(ema5))
        ema10lst.append(np.copy(ema10))
        ema20lst.append(np.copy(ema20))

    return np.array(ema5lst), np.array(ema10lst), np.array(ema20lst)

def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):

    layers = []

    nodes_increment = (last_layer_nodes - first_layer_nodes)/ (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment
    print(layers)
    return layers

def mlp_model(input_dims, output_dims, n_layers, first_layer_nodes, last_layer_nodes, activation_func, loss_func):

    lr = 1e-4
    optimizer = Adam(lr = lr, beta_1 = 0.5, beta_2 = 0.9)
    model = Sequential()
    n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
    for i in range(1, n_layers):
        if i==1:
            model.add(Dense(first_layer_nodes, input_dim=input_dims, kernel_regularizer=L2(0.1), activation=activation_func))
            model.add(BatchNormalization(momentum=0.8))
        else:
            model.add(Dense(n_nodes[i-1], kernel_regularizer=L2(0.1), activation=activation_func))
            model.add(BatchNormalization(momentum=0.8))

    #Finally, the output layer should have a single node in binary classification
    #model.add(Dense(output_dims, kernel_initializer='normal'))
    model.add(Dense(output_dims, kernel_initializer='normal', activation='sigmoid'))
    #model.compile(optimizer=optimizer, loss=loss_func, metrics = [MeanAbsoluteError(),
    #                                                              RootMeanSquaredError(),
    #                                                              MeanSquaredError()])
    model.compile(optimizer=optimizer, loss=loss_func, metrics = ['binary_accuracy'])
    return model

def FindPair(pred, sec_list):
    """
    Input:
        pred: 
         dimension: days*No. stock in sector
         meaning: possibility of a stock to be in the top half

        sec_list: 
         dimension: No. stock in sector
         meaning: security codes of stocks in sector
    """
    output, rest = [], []
    num_pair = len(sec_list)//2
    odd = bool(len(sec_list)%2)
    for p in pred:
        rank = np.argsort(p)[::-1]
        tmp = []
        for i in range(num_pair):
            tmp.append([p[rank[i]]-p[rank[-i-1]], sec_list[rank[i]], sec_list[rank[-i-1]]])
        if odd: rest.append([sec_list[rank[num_pair]]])
        output.append(tmp) 

    if not odd: 
        rest = None
    else:
        rest = np.array(rest)
    return np.array(output), rest

def WholeRank(pred, rest):
    """
    Input:
        pred: days*total pairs
        rest: days*total unpaired
  
    Output:
        output: days*total stocks (2000)
    """
    output = []
    for p, r in zip(pred, rest):
         p.tolist().sort(key=lambda x:x[0], reverse=True)
         head, tail = [_[1] for _ in p], [_[2] for _ in p][::-1]
         output.append(head+r.tolist()+tail)
    return np.array(output)
         

