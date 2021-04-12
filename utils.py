import numpy as np
import pandas as pd
import csv

def reduce_memory_usage(dataframe):
    start_mem_usg = dataframe.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in dataframe.columns:
        if dataframe[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("**********")
            print("Column: ",col)
            print("dtype before: ",dataframe[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = dataframe[col].max()
            mn = dataframe[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(dataframe[col]).all(): 
                NAlist.append(col)
                dataframe[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = dataframe[col].fillna(0).astype(np.int64)
            result = (dataframe[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        dataframe[col] = dataframe[col].astype(np.uint8)
                    elif mx < 65535:
                        dataframe[col] = dataframe[col].astype(np.uint16)
                    elif mx < 4294967295:
                        dataframe[col] = dataframe[col].astype(np.uint32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                dataframe[col] = dataframe[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",dataframe[col].dtype)
            print("**********")
    
    # Print final result
    print("__MEMORY USAGE AFTER COMPLETION:__")
    mem_usg = dataframe.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return dataframe, NAlist

def save_to_csv(data, path):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Id','Predicted'])
        for i in range(len(data)):
            writer.writerow([i,float(data[i])])