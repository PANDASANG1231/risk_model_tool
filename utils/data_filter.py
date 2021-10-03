import numpy as np
import pandas as pd

def gen_corr(df):
    
    df = df.loc[(df != -1).T.any(), :]
    
    def transform(x):
        if x == 1:
            return 0
        elif x < 0:
            return -x
        else:
            return x

    corr = df.corr()
    corr = corr.applymap(transform)
    return corr


def filt_corr(corr, thresh):
    
    for col in corr.columns:
        for idx in corr.index:
            try:
                if corr.loc[idx, col]>thresh:
                    corr.loc[idx, :] = -1
                    corr.loc[:, idx] = -1
            except:
                pass
    
    picked = (corr != -1).any()
    corr = corr.loc[picked, picked]
    return corr
