import numpy  as np
import pandas as pd

#------------------------------data preparation--------------------------------
"""
read data from path and store them in a np array
"""

def readDataToNp(loc, debug = False):
    with open(loc, 'r') as f:
        rawdata = f.readlines()
        if debug == True: 
            print('\n rawdata:', rawdata[:5], '\n len of rawdata: ', len(rawdata))
        dataSplited = []
        for row in rawdata:
            tmpstr = row.split('\n')[0].split(',')[:-1]
            tmplist = []
            for st in tmpstr:
                tmplist.append(st.strip())
            dataSplited.append(tmplist)
        if debug == True: 
            print('\n dataSplited:', dataSplited[:5])
        df = pd.DataFrame(dataSplited[1:], dtype = float)
        df.columns = dataSplited[0]
        
        #remove I features
        df1 = pd.DataFrame()
        df1 = df
        df1.drop(['F6', 'I1','I2','I3', 'I4', 'I5'], axis = 1, inplace=True)
#        df1 = df.iloc[:, :5]
#        df1['F6'] = df['F6']
#        df1['nuLES'] = df['nuLES']
        df = df1
        
        if debug == True: 
            print('\n df:', df.head(5))
        
        data = np.array(df, dtype = float)
        if debug == True: 
            print('\n data:', data[:5])
            
        #preprocessing
#        data[:, :-1] -= np.mean(data[:, :-1], axis = 0)
#        data[:, :-1] /= np.std(data[:, :-1],  axis = 0)
        mean = np.mean(data, axis = 0)
        std  = np.std(data,  axis = 0)
        data -= mean
        data /= std
        print('mean: ', np.mean(data, axis = 0))
        print(' std: ', np.std(data,  axis = 0))
#        data /= np.abs(np.max(data, axis = 0) - np.min(data, axis = 0))
#        data = (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis = 0))
    return data, mean, std, list(df.columns)
"""
split original data into train, test and validate set
"""

def prepareData(loc, p_train = 0.9, p_val = 0.1, debug = False):
    dataSet = {}
    
    data, mean, std, col_name = readDataToNp(loc, debug = debug)
    np.random.shuffle(data)
    #only use part of the data
#    data = data[:round(len(data)*0.8)]
    #subsample the data
    num_train = int(len(data) * p_train)
    num_val   = int(num_train * p_val)
    num_test  = len(data) - num_train
    
    
    mask_train = list(range(0, num_train))
    mask_val   = list(range(num_train - num_val, num_train))
    mask_test  = list(range(num_train, num_train + num_test))
    
    if debug == True:
        print(
        '\nwholeData len: {}'.format(len(mask_train) + len(mask_test)),
        '\nmask_train: {}-{}, len: {}'.format(mask_train[0], mask_train[-1], len(mask_train)),
        '\nmask_val: {}-{}, len: {}'.format(mask_val[0], mask_val[-1], len(mask_val)),
        '\nmask_test: {}-{}, len: {}'.format(mask_test[0], mask_test[-1], len(mask_test)))
        
    dataSet['X_train'] = data[mask_train, :-1]
    dataSet['y_train'] = data[mask_train,  -1]
    dataSet['X_val']   = data[mask_val, :-1]
    dataSet['y_val']   = data[mask_val,  -1]
    dataSet['X_test']  = data[mask_test, :-1]
    dataSet['y_test']  = data[mask_test,  -1]
    
    return dataSet, mean, std, col_name


def splitData(data, p_train = 0.9, p_val = 0.1, debug = False):
    dataSet = {}
    
#    data,_,_ = readDataToNp(loc)
    np.random.shuffle(data)
    #subsample the data
    num_train = int(len(data) * p_train)
    num_val   = int(num_train * p_val)
    num_test  = len(data) - num_train
    
    
    mask_train = list(range(0, num_train))
    mask_val   = list(range(num_train - num_val, num_train))
    mask_test  = list(range(num_train, num_train + num_test))
    
    if debug == True:
        print(
        '\nmask_train: {}-{}, len: {}'.format(mask_train[0], mask_train[-1], len(mask_train)),
        '\nmask_val: {}-{}, len: {}'.format(mask_val[0], mask_val[-1], len(mask_val)),
        '\nmask_test: {}-{}, len: {}'.format(mask_test[0], mask_test[-1], len(mask_test)))
        
    dataSet['X_train'] = data[mask_train, :-1]
    dataSet['y_train'] = data[mask_train,  -1]
    dataSet['X_val']   = data[mask_val, :-1]
    dataSet['y_val']   = data[mask_val,  -1]
    dataSet['X_test']  = data[mask_test, :-1]
    dataSet['y_test']  = data[mask_test,  -1]
    
    return dataSet   



