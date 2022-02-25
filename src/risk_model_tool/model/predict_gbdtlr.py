import os
import pickle
import numpy as np
import pandas as pd

def simple_appnamelist(x):
    for index,value in enumerate(x):
        x[index]=x[index].split('-')[0]
    return x

def trnasfer_input_lr(preds_leaf, num_leaves):
    preds_leaf = np.array(preds_leaf)
    transform_input = np.zeros([len(preds_leaf), len(preds_leaf[0]) * num_leaves], dtype=np.int64)  # N * num_tress * num_leafs
    for i in range(0, len(preds_leaf)):
        temp = np.arange(len(preds_leaf[0])) * num_leaves + np.array(preds_leaf[i])
        transform_input[i][temp] += 1
    return transform_input

def trnasfer_input_gbdt(userapp, dict_ref):
    appname_list=userapp.split("##")
    appname_list=simple_appnamelist(appname_list)
    appid_list = [dict_ref.get(var,max(dict_ref.values())) for var in appname_list]
    
    user_app_list=[0,]*(max(dict_ref.values())+1)
    for i in appid_list:
        user_app_list[i]+=1
    #print(sorted(appid_list))
    return np.array(user_app_list)


def gbdt_predict(user_app_list, gbm, pred_leaf=True):
    #print(user_app_list)
    #print(len(user_app_list))
    preds_leaf = gbm.predict(user_app_list, pred_leaf=pred_leaf)
    #print(preds_leaf)
    return preds_leaf

def lr_predict(preds_leaf, lm, num_leaves=64):
    transform_input = trnasfer_input_lr(preds_leaf, num_leaves)
    y_pred = lm.predict_proba(np.array(transform_input)) 
    return [x[1] for x in y_pred]


def gbdt_lr_predict(gbm, lm, dict_ref, input_dict, gbdt_only=False, NUM_LEAVES=64):
    
    def __call_trnasfer_input_gbdt__(userapp):
        
        return trnasfer_input_gbdt(userapp, dict_ref)
    
    if isinstance(input_dict, dict):
        df = pd.DataFrame([input_dict])
    elif isinstance(input_dict, list):
        df = pd.DataFrame(input_dict)
    else:
        df = input_dict
    
    lgbm_input = df["str_applist_app_name"].apply(__call_trnasfer_input_gbdt__)
    lgbm_input = np.array([xx for xx in lgbm_input])

    if gbdt_only==True:
        score = gbdt_predict(lgbm_input, gbm, pred_leaf=False)
        return score
    else:
        preds_leaf = gbdt_predict(lgbm_input, gbm, pred_leaf=True)
        score = lr_predict(preds_leaf, lm, NUM_LEAVES)
        return score 
