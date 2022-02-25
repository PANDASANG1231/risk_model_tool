import os
import sys
import time
import pandas as pd
sys.path.append('/opt/notebooks/ppd_gits/ML_Platform/')

import numpy as np
import lightgbm as lgb
import one_var


def filr(dataframe, threshold=70, idx="gain", col="feature"):
    filter_ = dataframe[idx] > threshold
    rst = dataframe.loc[filter_, col]
    return sorted([x for x in rst])
    
def fit(X_train, y_train, X_valid, y_valid, X_test, y_test, params, cate, weight):
    lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cate, weight=weight)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cate, reference=lgb_train)
    lgb_test = lgb.Dataset(X_test, label=y_test, categorical_feature=cate, reference=lgb_train)

    gbm = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid, lgb_test], valid_names=["train", "valid", "test"])
    return gbm

def evaluate(model, X_train, y_train,
             X_valid, y_valid, X_test, y_test):
    
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)

    ks0, auc0 = one_var.evaluate_performance(y_train,
                                             pred_train,                                     
                                             to_plot=False,
                                             to_report=False)

    ks1, auc1 = one_var.evaluate_performance(y_valid,
                                             pred_valid,
                                             to_plot=False,
                                             to_report=False)

    ks2, auc2 = one_var.evaluate_performance(y_test,
                                             pred_test,
                                             to_plot=False,
                                             to_report=False)
    
    return [(ks0, auc0), (ks1, auc1), (ks2, auc2)]

def cross_validation(x_ins_oos, y_ins_oos, x_oot, y_oot, kfold, params,
                     features_selected, target_name):
    
    imp_all = []
    perf_all = []
    
    x_ins_oos.reset_index(inplace=True)
    y_ins_oos.reset_index(inplace=True)
    X_train_valid = x_ins_oos[features_selected]
    y_train_valid = y_ins_oos
    X_test = x_oot[features_selected]
    y_test = y_oot[target_name]
    
    from sklearn.model_selection import KFold
    kf = KFold(kfold, shuffle=True, random_state=1)

    for i, (ins_idx, oos_idx) in enumerate(kf.split(X_train_valid)):
        
        print(' lgb kfold: {}  of  {} : '.format(i+1, kfold))
        X_train = X_train_valid.loc[ins_idx, :]
        X_valid = X_train_valid.loc[oos_idx, :] 
        y_train = y_train_valid.loc[ins_idx, target_name]
        y_valid = y_train_valid.loc[oos_idx, target_name]
        
        gbm = fit(X_train, y_train, X_valid, y_valid, X_test, y_test, params)
        rst = evaluate(gbm, X_train, y_train, X_valid,
                       y_valid, X_test, y_test)
        imp = pd.Series(gbm.feature_importance(), name='split').to_frame() \
                .assign(feature=gbm.feature_name()) \
                .assign(gain=gbm.feature_importance(importance_type='gain')) \
                .sort_values('split', ascending=False)
                
        perf_all.append(rst + [i])
        imp_all.append(imp)
        
    
    return imp_all, perf_all

def save(gbm, name, mdir="./"):
    gbm.save_model(os.path.join(mdir, name),num_iteration=-1)

def train(x_ins, y_ins, x_oos, y_oos, x_oot, y_oot, params,
          features_selected, target_name,cate='auto', weight=None):
    
    X_train = x_ins[features_selected]
    X_valid = x_oos[features_selected]
    X_test = x_oot[features_selected]
    y_train = y_ins[target_name]
    y_valid = y_oos[target_name]
    y_test = y_oot[target_name]
    
    gbm = fit(X_train, y_train, X_valid, y_valid, X_test, y_test, params, cate=cate, weight=weight)
    
    try:
        evaluate(gbm, X_train, y_train, X_valid,
                 y_valid, X_test, y_test)
    except Exception as err:
        print(err)
    
    imp = pd.Series(gbm.feature_importance(), name='split').to_frame() \
                   .assign(feature=gbm.feature_name()) \
                   .assign(gain=gbm.feature_importance(importance_type='gain')) \
                   .sort_values('split', ascending=False)
    
    if not os.path.exists("./tmp_model/"):
        os.mkdir("./tmp_model/")
    else:
        save(gbm, "tmp_{}".format(time.time()), "./tmp_model/")
            
    return gbm, imp 


def evaluate_by_target(gbm, data, target_col, num_iter, to_plot=False):
    y_test = data[target_col]
    pred_test = gbm.predict(data[gbm.feature_name()], num_iter)
    
    zipped = filter(lambda x: x[0]>=0, zip(y_test, pred_test))
    y_test, pred_test = zip(*zipped)
    
    ks0, auc0 = one_var.evaluate_performance(y_test,
                                         pred_test,                                    
                                         to_plot=to_plot,
                                         to_report=False)
    print(target_col, ":", [ks0, auc0])
    return ks0, auc0   


def load_model(model_file):
    gbm = lgb.Booster(model_file=model_file)

    imp = pd.Series(gbm.feature_importance(), name='split').to_frame() \
            .assign(feature=gbm.feature_name()) \
            .assign(gain=gbm.feature_importance(importance_type='gain')) 
    
    return gbm, imp


### Batch Evalueate Tools 20191009

def evalu_stat_by_model(gbm, data_oot, 
                        to_plot, to_report,
                        true_targets, pred_target):
    rst = []
    for target_name in true_targets:
        df_tmp = data_oot[["userid", target_name]]
        df_tmp["pred_{}".format(pred_target)] = gbm.predict(data_oot[gbm.feature_name()])
        df_tmp = df_tmp[df_tmp[target_name] != -1]
        cnt, rate = df_tmp.shape[0], df_tmp[target_name].mean()
        tmp =one_var.evaluate_performance(y_pred=df_tmp["pred_{}".format(pred_target)], 
                                          y_true=df_tmp[target_name], 
                                          to_plot=to_plot,
                                          to_report=to_report)
        
        rst.append([target_name, pred_target, cnt, rate] + list(tmp))

    rst = pd.DataFrame(rst, columns=["target_name", "pred_name",
                                     "cnt", "rate", "ks", "auc"])
    return rst
    
def evalu_stat_by_model(data_oot, to_plot, to_report,
                        true_targets, pred_score):
    rst = []
    for target_name in true_targets:
        df_tmp = data_oot[["userid", target_name, pred_score]]
        df_tmp = df_tmp[df_tmp[target_name] != -1]
        cnt, rate = df_tmp.shape[0], df_tmp[target_name].mean()
        try:
            tmp =one_var.evaluate_performance(y_pred=df_tmp[pred_score], 
                                          y_true=df_tmp[target_name], 
                                          to_plot=to_plot,
                                          to_report=to_report)
            
            rst.append([target_name, pred_score, cnt, rate] + list(tmp))
        except Exception as exp:
            print(exp)
        
        

    rst = pd.DataFrame(rst, columns=["target_name", "pred_name",
                                     "cnt", "rate", "ks", "auc"])
    return rst
    

def evalu_stat(data_oot,
               true_targets=["target_1m_30"],
               pred_scores=["pred_target_1m_30"],
               to_plot=False, to_report=False):
    
    rst = []
    for pred_score in pred_scores:
        tmp_df = evalu_stat_by_model(data_oot, to_plot, to_report,
                                     true_targets, pred_score)
        
        rst.append(tmp_df)
    
    return pd.concat(rst, axis=0)

### Evaluate Duerate by two_group of people

def sample_describe(df, amt_col, prin, title, creationdate):

    
    cnt = df.shape[0]
    prin_avg = df[prin].sum()/cnt

    df_tmp = df[df[amt_col] >= 0]
    seen_cnt = df_tmp.shape[0]

    date_max = df_tmp[creationdate].max()[0:10]
    date_min = df_tmp[creationdate].min()[0:10]
    
    tg_rt = df_tmp[amt_col].sum()/df_tmp[prin].sum()

    rst = pd.DataFrame([[cnt, seen_cnt, "{}~{}".format(date_min, date_max), tg_rt, prin_avg, title]], columns=["count", "seen_count",  "creationdate", "amount_rate", "prin_sum", "note"])

    return rst


def compare_people(df_1, df_2,
                   userid = "userid",
                   amt_col = "amount_1m_30",
                   prin = 'principal',
                   title=["df_1", "df_2"]):
    

    def sample_describe(df, amt_col, prin, title):

        cnt = df.shape[0]
        prin_avg = df[prin].sum()/cnt
        
        df_tmp = df[df[amt_col] >= 0]
        seen_cnt = df_tmp.shape[0]

        tg_rt = df_tmp[amt_col].sum()/df_tmp[prin].sum()

        rst = pd.DataFrame([[cnt, seen_cnt, tg_rt, prin_avg, title]], columns=["count", "seen_count", "amount_rate", "prin_sum", "note"])

        return rst
    
    rst = []
    dfs = [df_1, df_2, pd.merge(df_1, df_2[[userid]])]
    title += ["both in"]
    for i, j in zip(dfs, title):
        
        rst.append(sample_describe(i, amt_col, prin, j))
        
    return pd.concat(rst)

def compare_by_confirm_rate(simu_df, main_score, cur_score, p_compare, p_confirm=0.8, amt_col = "amount_1m_30"):
    
    simu_df_tmp = simu_df[simu_df[amt_col]>=0]
    
    simu_df_tmp.sort_values(main_score, ascending=True, inplace=True)
    simu_df_tmp = simu_df_tmp.head(int(simu_df_tmp.shape[0]*p_confirm))

    simu_df_tmp_bymain = simu_df_tmp.tail(int(simu_df_tmp.shape[0]*p_compare))
    simu_df_tmp_bycur = simu_df_tmp.sort_values(cur_score, ascending=True).tail(int(simu_df_tmp.shape[0]*p_compare))
    
    rst = compare_people(simu_df_tmp_bymain, simu_df_tmp_bycur, amt_col=amt_col, title=["main_score", "aid_score"])
    
    return rst



def compare_by_confirm_rate_fg(simu_df, main_score, cur_score, p_compare, p_confirm=1.0, amt_col = "amount_1m_30"):
    
    simu_df_tmp = simu_df[simu_df[amt_col]>=0]
    
    simu_df_tmp.sort_values(main_score, ascending=True, inplace=True)
    simu_df_tmp = simu_df_tmp.head(int(simu_df_tmp.shape[0]*p_confirm))

    simu_df_tmp_bymain = simu_df_tmp.head(int(simu_df_tmp.shape[0]*p_compare))
    simu_df_tmp_bycur = simu_df_tmp.sort_values(cur_score, ascending=True).head(int(simu_df_tmp.shape[0]*p_compare))
    
    rst = compare_people(simu_df_tmp_bymain, simu_df_tmp_bycur, amt_col=amt_col, title=["main_score", "aid_score"])
    
    return rst





