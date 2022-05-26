import os
import csv
import collections
import pandas as pd

sector = {}
with open('/home/fs01/yy692/Kaggle/Kaggle_TokyoStock/data/stock_list.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        sector[row[0]] = row[7]

sec_list = os.listdir('train_set')
#sec_list = os.listdir('validation_set')

sector_list = collections.defaultdict(list)
for sec_code in sec_list:
    sector_list[sector[sec_code[:-4]]].append(sec_code)

for key, lst in sector_list.items():
    feature = None
    target = None
    for sec_code in lst:
        df = pd.read_csv('train_set/'+sec_code)
        #df = pd.read_csv('validation_set/'+sec_code)
        if feature is None:
            feature = df[['Return_Close', 'Return_Open', 'Return_High', 'Return_Low', 'Volume']]
            target = df['Target']
        else:
            feature = pd.concat([feature, df[['Return_Close', 'Return_Open', 'Return_High', 'Return_Low', 'Volume']]], axis=1)
            target = pd.concat([target, df['Target']], axis=1)
    feature.to_csv('train_x/sector_x_'+key+'.csv')
    target.to_csv('train_y/sector_y_'+key+'.csv')
    #feature.to_csv('validation_x/sector_x_'+key+'.csv')
    #target.to_csv('validation_y/sector_y_'+key+'.csv')


#for key, lst in sector_list.items():
#    print(key,lst)
