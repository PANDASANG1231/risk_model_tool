## -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import lightgbm as lgb
from matplotlib.ticker import FuncFormatter, LogitLocator, FixedLocator
from lgbm_util import search_split_with_feature, quick_model
from sklearn.model_selection import train_test_split
from pandas.api.types import is_categorical

def evaluate_performance(
        y_true,
        y_pred,
        to_plot=True,
        to_report=True,
        binnum=10,
        figsize=None,
        accurate_confusion_matrix=False):
    '''
    评估模型的性能。
    在默认参数下会画三张图（to_plot = True）：
        第一张是 KS和AUC
        第二张是 分数分布 和 平均分数
        第三张是在percentile下的平均预测分和平均真实值
    在默认参数（to_report = True）下还会打印 KS，AUC 和 样本数量N，
        以及用全体样本的Target Rate做阈值时模型的分类表现和混淆矩阵。

    Args:
        y_true 真实的标签
        y_pred 预测的分数
        to_plot 是否画图。默认为 True
        to_report 是否打印报告。默认为 True,
        binnum 分bin的数量，默认为 10,
        figsize 图的大小，默认为 None,
        accurate_confusion_matrix 是否计算精准的混淆矩阵，耗时较长，否则用roc上的点计算，默认为 Fasle。

    Return:
        ks, roc_auc

    '''

    if isinstance(y_true, pd.core.series.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.core.series.Series):
        y_pred = y_pred.values

    if figsize is None:
        figsize = (20, 6)
    length = len(y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    target_rate = np.mean(y_true)
    positive_number = np.sum(y_true)
    mean_pred = np.mean(y_pred)

    ks = np.max(tpr - fpr)
    ks_ind = np.argmax(tpr - fpr)
    ks_tpr, ks_fpr = tpr[ks_ind], fpr[ks_ind]

    tr_ind = np.argmin(abs(thresholds - np.array(target_rate)))
    tpr_tr, fpr_tr = tpr[tr_ind], fpr[tr_ind]

    if to_report:
        print('KS=%.5f, AUC=%.5f, N=%d' % (ks, roc_auc, length))
        print(
            'With threshold %.5f which is the target rate, the classification report is as follow:' %
            (target_rate))
        print(classification_report(y_true, y_pred > target_rate))
        # TODO This part can be removed for we now understand where the error comes. we can avoid these problems.
        if not accurate_confusion_matrix:
            print(pd.DataFrame([[(1 - fpr_tr) * (length - positive_number),
                                 fpr_tr * (length - positive_number)],
                                [(1 - tpr_tr) * positive_number,
                                 tpr_tr * positive_number]],
                               index=['True 0', 'True 1'],
                               columns=['Pred 0', 'Pred 1'],
                               dtype=np.int))
        else:
            print(
                pd.DataFrame(
                    confusion_matrix(y_true, y_pred > target_rate),
                    index=['True 0', 'True 1'],
                    columns=['Pred 0', 'Pred 1']))

    if to_plot:
        # KS plot
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        plt.title('KS=%.3f AUC=%.3f' % (ks, roc_auc), fontsize=20)
        plt.plot([ks_fpr, ks_fpr], [ks_fpr, ks_tpr],
                 linewidth=3, color='r', label='KS')
        plt.plot(
            [fpr_tr],
            [tpr_tr],
            'k.',
            markersize=10,
            label='Target Rate Point')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive', fontsize=20)
        plt.ylabel('True Positive', fontsize=20)
        plt.legend()

        plt.subplot(1, 3, 2)

        plt.hist(y_pred, bins=binnum * 2)
        plt.axvline(
            x=target_rate,
            linestyle='--',
            color='r',
            label='Mean of Truth')
        plt.axvline(
            x=mean_pred,
            linestyle=':',
            color='b',
            label='Mean of Prediction')
        plt.title('N=%d Tru=%.3f Pred=%.3f' % (length, target_rate, mean_pred),
                  fontsize=20)
        plt.legend()
        plt.xlabel('Prediction', fontsize=20)
        plt.ylabel('Target Count', fontsize=20)

        ave_predict = []
        ave_target = []
        indices = np.argsort(y_pred)

        indexs = [int(round(length * i / binnum)) for i in range(binnum + 1)]

        for startind, endind in zip(indexs[:-1], indexs[1:]):
            ave_predict.append(np.mean(np.array(y_pred)[np.array(indices)[startind:endind]]))
            ave_target.append(np.mean(np.array(y_true)[np.array(indices)[startind:endind]]))

        plt.subplot(1, 3, 3)
        plt.plot(ave_predict, 'b.-', label='Prediction', markersize=5)
        plt.plot(ave_target, 'r.-', label='Truth', markersize=5)
        plt.legend()
        plt.xlabel('Percentile', fontsize=20)
        plt.ylabel('Target Rate / Prediction', fontsize=20)

        plt.show()

    return ks, roc_auc

def pd_date_string_2_timestamp(date_string_series):
    return pd.to_datetime(date_string_series).apply(lambda x : x.timestamp())

def _get_df_in_bins(se_x, se_y, bins):
    '''
    get the pos and neg count in the bins.
    '''
    pos_se = se_x[se_y == 1]
    neg_se = se_x[se_y == 0]
    pos = pos_se.value_counts(bins=bins).sort_index()
    neg = neg_se.value_counts(bins=bins).sort_index()

    if (bins is not None) and (np.isnan(se_x).sum() > 0):
        pos = pos.append(pd.Series([np.isnan(pos_se).sum()]))
        neg = neg.append(pd.Series([np.isnan(neg_se).sum()]))

    bins = pd.DataFrame.from_dict({'pos': pos, 'neg': neg})
    bins['count'] = bins['pos'] + bins['neg']
    return bins


def _calculate_iv(bins):
    '''
    calculate iv based on information given by the `bins` DataFrame.
    '''
    pn_sums = bins.sum()
    bin_count = bins['count']
    bins['p_data'] = bins['pos'] / bin_count
    bins['pos_p'] = (bin_count * bins['p']) / pn_sums['pos']
    bins['neg_p'] = (bin_count - bin_count * bins['p']) / pn_sums['neg']
    base_woe = np.log(pn_sums['pos'] / pn_sums['neg'])
    bins['woe'] = np.log(bins['p'] / (1 - bins['p'])) - base_woe
    iv = ((bins['pos_p'] - bins['neg_p']) * bins['woe']).sum()
    return iv, base_woe


def auto_xlog_bins(x, xlog=None):
    low_bound = x.min()
    high_bound = x.max()
    if xlog is None:
        xlog = False
        if low_bound >= 0:
            mean_point = (x.mean() - low_bound) / \
                (high_bound - low_bound)
            if mean_point < 0.2:
                xlog = True

    if xlog:
        if low_bound == 0:
            if high_bound <= 1:
                bins = np.linspace(low_bound, high_bound, 50)
                xscale = 'linear'
            else:
                bins_head = np.linspace(0, 1, 10)
                bins_tail = np.logspace(0, np.log10(high_bound), 50)
                bins = np.concatenate((bins_head, bins_tail[1:]))
                xscale = 'symlog'
        else:
            bins = np.logspace(np.log10(low_bound), np.log10(high_bound), 50)
            xscale = 'log'
    else:
        bins = np.linspace(low_bound, high_bound, 50)
        xscale = 'linear'

    return xlog, bins, xscale

def one_var_analysis(
        x_single,
        y_true,
        sub_range=None,
        xlog=None,
        woe_scale=False,
        more_smooth=True,
        rev_mapper=None,
        to_plot=True):
    '''
    单变量与标签间的分析，依赖LightGBM分bin以及拟合P_pos和P_neg。
    Args:
        x_single 单个x变量
        y_true 真实标签
        sub_range 在画图时可以传入一个tuple (from, to)来指定要画的区间 。默认为 None，画全部。
        xlog 在画图时是否在x轴上使用对数轴。默认为 None, 表示自适应。
        woe_scale，在画图时y是否使用woe，默认为False，使用 p 概率来作为绘图时的y。

    Return:
        gbm, lightGBM 模型
        df_pn_bins, 分bin情况以及各个bin的woe等属性。
        iv, IV 值
    '''

    iscat = is_categorical(x_single)
    pd_x = x_single.to_frame()

    lgb_train = lgb.Dataset(pd_x, y_true)

    num_leaves = 4 if iscat else (2 if more_smooth else 3)

    gbm = quick_model(
        x_single,
        y_true,
        num_leaves=num_leaves)

    model = gbm.dump_model()

    if not iscat:
        bins = search_split_with_feature(model)
        bins = sorted(set([-1e300] + bins + [1e300]))
    else:
        bins = None

    df_pn_bins = _get_df_in_bins(x_single, y_true, bins)

    if iscat:
        woe_x_df = df_pn_bins.index
    else:
        woe_x_df = df_pn_bins.index.map(
            lambda x: x.mid if isinstance(
                x, pd.Interval) else np.nan).T

    woe_x = pd.DataFrame(
        woe_x_df,
        columns=pd_x.columns)

    y_woe = gbm.predict(woe_x)
    df_pn_bins['p'] = y_woe
    evaluate_performance(y_true, gbm.predict(pd_x), to_plot=False)
    iv, base_woe = _calculate_iv(df_pn_bins)
    mean_p = (df_pn_bins['p'] * df_pn_bins['count']).sum() / \
        df_pn_bins['count'].sum()
    target_rate = df_pn_bins['pos'].sum() / df_pn_bins['count'].sum()

    print(
        'IV is %f; Mean p is %f; Target Rate is %f; Diff is %f (%f%%).' %
        (iv,
         mean_p,
         target_rate,
         mean_p - target_rate,
         (mean_p - target_rate) / target_rate * 100))

    if (not iscat) and 0 in df_pn_bins.index:
        print('The data contains NaN and its props are: Count: {count:.0f}, p: {p:.4f}, WoE: {woe:.4f}.'.format(
            **df_pn_bins.loc[0, ['count', 'p', 'woe']].to_dict()))

    if iscat and (rev_mapper is not None):
        df_pn_bins['cat_name'] = df_pn_bins.index.map(
            lambda x: ','.join(rev_mapper[x]) if x in rev_mapper else 'None')

    if iscat or (not to_plot):
        return gbm, df_pn_bins, iv

    x_single_u = pd.Series(x_single.unique())

    if sub_range is not None:
        low, high = sub_range
        assert low < high, 'In sub_range 1st value must be smaller than 2nd value.'
        x_single_u = x_single_u[(x_single_u >= low) & (x_single_u <= high)]
        x_single = x_single[(x_single >= low) & (x_single <= high)]

    pd_x_e = x_single_u.to_frame()
    y_pred = gbm.predict(pd_x_e)

    xlog, bins, xscale = auto_xlog_bins(x_single, xlog)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    plt.title('%s: IV=%.3f' % (x_single.name, iv), fontsize=20)

    ax1.hist(x_single[~np.isnan(x_single)], bins)
    ax2 = ax1.twinx()
    if woe_scale:
        y_pred = np.log(y_pred / (1 - y_pred)) - base_woe

    ax2.plot(x_single_u, y_pred, 'r.', alpha=0.5)
    ax2.yaxis.grid()
    ax1.set_xlabel('x value', fontsize=20)
    ax1.set_ylabel('count', fontsize=20)
    if woe_scale:
        ax2.set_ylabel('Weight Of Evidence', fontsize=20)
    else:
        ax2.set_ylabel('Smoothed Target Rate', fontsize=20)
    if xlog:
        plt.xscale(xscale)
    fig.tight_layout()
    plt.show()
    return gbm, df_pn_bins, iv

def apply_woe(woe_report, series):
    '''
    一个将计算好的WOE应用到变量上的工具
    '''
    if isinstance(woe_report.index, (pd.core.indexes.interval.IntervalIndex, pd.core.indexes.category.CategoricalIndex)):
        return woe_report.loc[series, 'woe'].as_matrix()
    else:
        report_no_na = woe_report.iloc[:-1, woe_report.columns.get_loc('woe')]
        report_no_na.index = pd.IntervalIndex(report_no_na.index)
        na_woe = woe_report.iloc[-1, woe_report.columns.get_loc('woe')]
        na_p = np.isnan(series).astype(float)
        return report_no_na[series].as_matrix() * (1 - na_p) + na_woe * na_p
        

def plot_two_distribution(ser_old, ser_new, xlog=None, sub_range=None):
    '''
    Args:
        ser_old 第一个要比较的变量
        ser_new 第二个要比较的变量
        xlog 在画图时是否在x轴上使用对数轴。默认为 None, 表示自适应。
        sub_range 在画图时可以传入一个tuple (from, to)来指定要画的区间 。默认为 None，画全部。
    '''
    name = ser_old.name if ser_old.name == ser_new.name else '%s:%s' % (
        ser_old.name, ser_new.name)
    ser_old = ser_old[~np.isnan(ser_old)]
    ser_new = ser_new[~np.isnan(ser_new)]
    if sub_range is not None:
        low, high = sub_range
        assert low < high, 'In sub_range 1st value must be smaller than 2nd value.'
        ser_old = ser_old[(ser_old >= low) & (ser_old <= high)]
        ser_new = ser_new[(ser_new >= low) & (ser_new <= high)]

    print('%s: #old: %d; #new: %d' % (name, len(ser_old), len(ser_new)))

    xlog, bins, xscale = auto_xlog_bins(pd.concat([ser_old, ser_new]), xlog)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ser_old, bins, alpha=0.5, label='old')
    ax.hist(ser_new, bins, alpha=0.5, label='new')
    if xlog:
        plt.xscale(xscale, basex=4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: x))
    ax.legend()
    plt.show()

    
def pivot_table_smooth(data, 
                   values=None, 
                   index=None, 
                   columns=None,
                   weight=None,
                   ):
    
    index = index if isinstance(index, list) else [index]
    columns = columns if isinstance(columns, list) else [columns]
    df_xy = pd.DataFrame(data[index + columns])

    gbm = quick_model(df_xy, data[values], num_leaves=3, weight=data[weight] if weight is not None else None)    
    df_xy['y_pred'] = gbm.predict(df_xy)
    
    return pd.pivot_table(df_xy, values='y_pred', index=index,
                  columns=columns, aggfunc=np.mean)

def digitize_by_percentile(series, bin_num=10):
    bins = set(np.percentile(series,list(range(0, 101, 100//bin_num))))
    bins = sorted(list(bins))
    bins[0] -= 1
    bins[-1] += 1    
    return np.digitize(series, bins)
