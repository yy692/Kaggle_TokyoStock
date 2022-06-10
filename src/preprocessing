import pandas as pd
import numpy as np
import os
import collections

def separate_stocks(read_path, save_path):
    '''
    Input:
        read_path (str) where to read train/supplemental files
        save_path (str) where to save the loaded price info for every stock
    Output:
        security_code_dict (dictionary) security code -> number of trading days
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    df = pd.read_csv(read_path)
    df.drop(labels=['AdjustmentFactor','ExpectedDividend', 'SupervisionFlag'], axis=1, inplace=True)
    df.drop(labels = ['RowId', 'Open', 'High', 'Low', 'Volume'], axis=1, inplace=True)
    df.sort_values(['SecuritiesCode', 'Date'], inplace = True)
    
    security_code_dict = {}
    longest_days = 0
    length = df.shape[0]
    start = 0
    end = 0
    security_code = df.iloc[start]['SecuritiesCode']
    for i in range(length):
        if df.iloc[i]['SecuritiesCode']!=security_code:
            end = i
            security_code_dict[security_code] = end-start
            temp_df = pd.DataFrame(df.iloc[start:end])
            temp_df.to_csv(save_path+str(security_code)+'.csv', index = False)
            start = i
            security_code = df.iloc[start]['SecuritiesCode']

    security_code_dict[security_code] = length-start
    temp_df = pd.DataFrame(df.iloc[start:])
    temp_df.to_csv(save_path+str(security_code)+'.csv', index = False)
    
    return security_code_dict

def feature_processing(save_path, security_code_dict):
    '''
    Input:
        path: (str) folder where the raw data (for each stock) are stored
    '''
    for security_code in security_code_dict.keys():
        df = pd.read_csv(save_path+str(security_code)+'.csv')
        df['pClose'] = df['Close'].shift(1)
        df['Return_Close'] = (df['Close'] - df['pClose'])/df['pClose'] 
        df.fillna(0, inplace = True)
        df.to_csv(save_path+str(security_code)+'.csv', index=False)

def fill_date(file_name, template_df):
    '''
    Input:
        file_name: (str) path to the file that need to be processed
        template_df: (dataframe) longest stock info dataframe
    '''
    temp_df = template_df.copy(deep=True)
    df = pd.read_csv(file_name)
    len_df = df.shape[0]
    p1 = 0
    p2 = 0
    while(p1<temp_df.shape[0] and p2<len_df):
        if df.iloc[p2]['Date'] == temp_df.iloc[p1]['Date']:
            temp_df.iloc[p1, df.columns.get_loc('Return_Close')] = df.iloc[p2]['Return_Close']
            #temp_df.iloc[p1, df.columns.get_loc('Return_Open')] = df.iloc[p2]['Return_Open']
            #temp_df.iloc[p1, df.columns.get_loc('Return_High')] = df.iloc[p2]['Return_High']
            #temp_df.iloc[p1, df.columns.get_loc('Return_Low')] = df.iloc[p2]['Return_Low']
            temp_df.iloc[p1, df.columns.get_loc('Target')] = df.iloc[p2]['Target']
            #temp_df.iloc[p1, df.columns.get_loc('Volume')] = df.iloc[p2]['Volume']
            p1+=1
            p2+=1
        else:
            temp_df.iloc[p1, df.columns.get_loc('Return_Close')] = 0
            #temp_df.iloc[p1, df.columns.get_loc('Return_Open')] = 0
            #temp_df.iloc[p1, df.columns.get_loc('Return_High')] = 0
            #temp_df.iloc[p1, df.columns.get_loc('Return_Low')] = 0
            temp_df.iloc[p1, df.columns.get_loc('Target')] = 0
            #temp_df.iloc[p1, df.columns.get_loc('Volume')] = 0
            p1+=1
    while(p1<temp_df.shape[0]):
        temp_df.iloc[p1, df.columns.get_loc('Return_Close')] = 0
        #temp_df.iloc[p1, df.columns.get_loc('Return_Open')] = 0
        #temp_df.iloc[p1, df.columns.get_loc('Return_High')] = 0
        #temp_df.iloc[p1, df.columns.get_loc('Return_Low')] = 0
        temp_df.iloc[p1, df.columns.get_loc('Target')] = 0
        #temp_df.iloc[p1, df.columns.get_loc('Volume')] = 0
        p1+=1
    temp_df.to_csv(file_name, index = False)

def fill_date_batch(path, security_code_dict):
    '''
        Input:
            path (str) folder where the processed (for each stock) are stored 
            security_code_dict (dictionary) security code -> number of trading days
    '''
    longest_stock = max(list(security_code_dict.values()))
    stocks_need_filling = []
    for key, val in security_code_dict.items():
        if val < longest_stock: stocks_need_filling.append(key)
        if val == longest_stock: template_stock = key

    template_df = pd.read_csv(path+str(template_stock)+'.csv')
    for security_code in stocks_need_filling:
        file_name = path+str(security_code)+'.csv'
        fill_date(file_name, template_df)

def concate_train_and_sup_files(train_path, sup_path, all_path, security_code_dict):
    for secu_code in security_code_dict.keys():
        train_df = pd.read_csv(train_path+str(secu_code)+'.csv')
        sup_df = pd.read_csv(sup_path+str(secu_code)+'.csv')
        all_df = pd.concat([train_df, sup_df], axis=0)
        all_df.to_csv(all_path+str(secu_code)+'.csv',index = False)

def group_by_sectors(stock_list_path, data_path, write_path, sector_path, isTraining):
    '''
        stock_list_path (str) stock_list.csv path
        data_path (str) stock data path
        write_path (str) output directory
        sector_path (str) directory to save sector information (security codes within a sector)
        isTraining (bool) is this processing the traning data
    '''
    sector = {}
    with open(stock_list_path+'stock_list.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            sector[row[0]] = row[7]
    
    sec_list = os.listdir(data_path)
    
    sector_list = collections.defaultdict(list)
    for sec_code in sec_list:
        sector_list[sector[sec_code[:-4]]].append(sec_codei[:-4])
    
    for key, lst in sector_list.items():
        lst.sort(key = lambda x: int(x))
        feature = None
        target = None
        for sec_code in lst:
            df = pd.read_csv(data_path+sec_code+'.csv')
            if feature is None:
                #feature = df[['Return_Close', 'Return_Open', 'Return_High', 'Return_Low', 'Volume']]
                feature = df[['Return_Close']]
                if isTraining: target = df['Target']
            else:
                #feature = pd.concat([feature, df[['Return_Close', 'Return_Open', 'Return_High', 'Return_Low', 'Volume']]], axis=1)
                feature = pd.concat([feature, df[['Return_Close']]], axis=1)
                if isTraining: target = pd.concat([target, df['Target']], axis=1)
        feature.to_csv(write_path+'sector_x_'+key+'.csv', index=False)
        if isTraining: target.to_csv(write_path+'sector_y_'+key+'.csv', index=False)
      
        pd.DataFrame(lst).to_csv(sector_path+str(key)+'.csv', index=False, header=False)
        
        
def preprocessing():
    train_path = '/kaggle/input/jpx-tokyo-stock-exchange-prediction/train_files/stock_prices.csv'
    sup_path = '/kaggle/input/jpx-tokyo-stock-exchange-prediction/supplemental_files/stock_prices.csv'
    train_save_path = '/kaggle/working/train/'+ 'stocks/'
    sup_save_path = '/kaggle/working/sup/' + 'stocks/'
    stock_list_path = '/kaggle/input/jpx-tokyo-stock-exchange-prediction/stock_list.csv'
    
    all_path = '/kaggle/working/all/'
    all_stock_path = '/kaggle/working/all/' + 'stocks/'
    all_sector_path = '/kaggle/working/all/' + 'sectors/'
    
    sector_path = '/kaggle/working/sector_list/'

    if not os.path.exists(train_save_path): os.makedirs(train_save_path)
    if not os.path.exists(sup_save_path): os.makedirs(sup_save_path)

    if not os.path.exists(all_path): os.makedirs(all_path)
    if not os.path.exists(sector_path): os.makedirs(sector_path)

    if not os.path.exists(all_stock_path): os.makedirs(all_stock_path)
    if not os.path.exists(all_sector_path): os.makedirs(all_sector_path)

    train_security_code_dict = separate_stocks(train_path, train_save_path)
    sup_security_code_dict = separate_stocks(sup_path, sup_save_path)

    feature_processing(train_save_path, train_security_code_dict)
    feature_processing(sup_save_path, sup_security_code_dict)

    fill_date_batch(train_save_path, train_security_code_dict)
    fill_date_batch(sup_save_path, sup_security_code_dict)
    concate_train_and_sup_files(train_save_path, sup_save_path, all_stock_path, train_security_code_dict)

    group_by_sectors(stock_list_path, all_stock_path, all_sector_path, sector_path, True)
