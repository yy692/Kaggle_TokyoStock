import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import jpx_utils
import jpx_preprocessing
import tensorflow.keras.backend as kb
import tensorflow as tf
print('import success!')

def run_MLP(xfile, yfile, sec_code_file, sector, window=5):
    '''
    Input:
        xfile: (str) path to dataset input feature
        yfile: (str) path to dataset target
        sec_code_file: (str) path to the file where stock security numbers are stored
        sector: (str) sector where the model is trained
        window (str): (int) lenght of time window
    Output:
        yPr: (np array) test days*number of stocks
        sec_code_list: (np array of str) security numbers of the stocks
    '''
    offset = window-1
    xTr, yTr, sec_code_list = jpx_utils.dataloader(xfile, yfile, sec_code_file)
    num_stock = len(sec_code_list)
    ema5Tr, ema10Tr, ema20Tr = jpx_utils.MovingAverage(xTr)
    xTr, yTr = jpx_utils.FormTimeWindow(xTr, yTr, window)
    xTr = np.concatenate((xTr, ema5Tr[offset:], ema10Tr[offset:], ema20Tr[offset:]), axis=1)
    xVa, yVa = xTr[-60:], yTr[-60:]
    xTr, yTr = xTr[:-60], yTr[:-60]
    
    #print(xTr.shape, yTr.shape, xTe.shape, yTe.shape, sec_code_list.shape)

    #mlp__first_layer_nodes = [xTr.shape[1]*2],
    if os.path.exists('working'): return xVa, yVa
    model = jpx_utils.mlp_model(input_dims=xTr.shape[1],
                    output_dims=yTr.shape[1],
                    n_layers=10,
                    first_layer_nodes=min(num_stock*4*window,10000),
                    last_layer_nodes=num_stock*2,
                    activation_func='relu',
                    loss_func='binary_crossentropy')
    #print(model.summary())
    history = model.fit(xTr, yTr, batch_size = 50, epochs = 40, verbose = 0, shuffle=True)
    model.save('/home/fs01/yy692/Kaggle/Kaggle_TokyoStock/working/models/'+sector+'_model')

    kb.clear_session()
    return xVa, yVa

def RunValidation(xVa_lst, yVa_df, sector_stock_mapping):
    eps = 1e-5
    weights = np.array([1.0+(1.0/199.0)*i for i in range(200)][::-1])
    relt, rest_list = None, None
    for idx, xVa in enumerate(xVa_lst):
        model = tf.keras.models.load_model('/home/fs01/yy692/Kaggle/Kaggle_TokyoStock/working/models/'+str(idx+1)+'_model')
        yPr = model.predict(xVa)

        output, rest = jpx_utils.FindPair(yPr, sector_stock_mapping[idx])
        if relt is None:
            relt = output
        else:
            relt = np.concatenate((relt,output), axis=1)

        if rest is None: continue
        if rest_list is None:
            rest_list = rest
        else:
            rest_list = np.concatenate((rest_list, rest), axis=1)

    rank = jpx_utils.WholeRank(relt, rest_list)
    gains = []
    for day in range(len(rank)):
        topGain = np.sum(yVa_df.iloc[day].loc[rank[day][:200]].to_numpy()*weights)
        botGain = np.sum(yVa_df.iloc[day].loc[rank[day][-200:]].to_numpy()*np.flip(weights))
        gains.append(topGain - botGain)
    gains = np.array(gains)

    return np.mean(gains)/(np.std(gains)+eps)
 

if __name__ == "__main__":
    if not os.path.exists('all'): jpx_preprocessing.preprocessing()

    sectors = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    sector_path = '/home/fs01/yy692/Kaggle/Kaggle_TokyoStock/working/sector_list/'
    path = '/home/fs01/yy692/Kaggle/Kaggle_TokyoStock/working/all/sectors/'
    sector_stock_mapping = []
    
    for sec in sectors:
        sec_code_file = sector_path+sec+".csv"
        sec_code_list = pd.read_csv(sec_code_file, header = None, dtype=str).to_numpy()
        sector_stock_mapping.append(np.squeeze(sec_code_list))
    
    xVa_lst, yVa_df = [], None
    for sec in sectors:
        xfile = path+'sector_x_'+sec+'.csv'
        yfile = path+'sector_y_'+sec+'.csv'
        sec_code_file = sector_path+sec+".csv"
        xVa, yVa = run_MLP(xfile, yfile, sec_code_file, sec, window=5)
        xVa_lst.append(xVa.copy())

        tmp_df = pd.DataFrame(yVa)
        tmp_df.columns = sector_stock_mapping[int(sec)-1]
        if yVa_df is None: 
            yVa_df = tmp_df.copy()
        else:
            yVa_df = pd.concat([yVa_df, tmp_df.copy()], axis=1)
        #yVa_lst.append(yVa.copy())

    print(RunValidation(xVa_lst, yVa_df, sector_stock_mapping))
