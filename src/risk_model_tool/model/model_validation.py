# coding: utf-8


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def cut_bins(x, bins=10, method='equal'):
    """
    对x进行分bin，返回每个样本的bin值和每个bin的下界
    Parameters
    ----------
    x: numpy.ndarray or pandas.Series
        变量
    bins: int or list, default 10
        分bin的个数
    method: str, default 'equal pop', options ['equal', 'quantile', 'point']
        分bin方式，'equal'是等样本量分bin，'quantile'为使用分位点分bin, 'point'为使用指定的分位点分bin

    Returns
    -------
    bin_no: numpy.ndarray
        每个样本的bin值
    """
    if method not in ('equal', 'quantile', 'point'):
        raise ValueError('method only choose "quantile" or "point"')

    if method == 'equal':
        if type(bins) != int:
            raise ValueError('when choose "point" method, bins need int number')
        bin_no = pd.qcut(x, q=bins, labels=range(1, bins + 1), precision=10).astype(int)
    elif method == 'quantile':
        if type(bins) not in (list, np.ndarray):
            raise ValueError('when choose "quantile" method, bins need list or np.ndarray type')
        bin_no = pd.qcut(x, q=bins, labels=range(1, len(bins) + 1), precision=10).astype(int)
    elif method == 'point':
        if type(bins) not in (list, np.ndarray):
            raise ValueError('when choose "point" method, bins need list or np.ndarray type')
        bin_no = np.digitize(x, bins=bins, right=False)

    return bin_no


def show_model_performance(truth, predict, bins=10, title=None, save_path=None):
    """
    计算模型的TPR、FPR，绘制ROC曲线、TPR-FPR曲线和Sloping曲线
    Parameters
    ----------
    truth: numpy.ndarray
        样本的真实标签
    predict: numpy.ndarray
        样本的预测分数
    bins: int, default 10
        分bin个数
    title: str, default None
        图片名称，通常以数据集命名，eg. ins、oos、oot
    save_path: str, default None
        图片存储路径

    Returns
    -------
    df_sloping: DataFrame
        每个bin的target rate和模型分均值
    auc: float
        模型AUC值
    ks: float
        模型ks值
    """
    n = truth.size  # 样本量
    fpr, tpr, thresholds = roc_curve(truth, predict)
    diff = tpr - fpr
    auc_value = auc(fpr, tpr)
    ks = diff.max()
    maxidx = 1.0 * diff.argmax() / diff.size
    cut_point = thresholds[diff.argmax()]
    reject_porp = round(100.0 * (predict >= cut_point).sum() / predict.shape[0], 2)

    df_tmp = pd.DataFrame({'truth': truth, 'predict': predict})
    df_tmp['bin'] = cut_bins(df_tmp['predict'], bins=bins, method='equal')
    group = df_tmp.groupby('bin')
    df_sloping = group['truth'].count().to_frame().reset_index(drop=False)
    df_sloping.columns = ['bin', 'sample_count']
    df_sloping['target_rate'] = group['truth'].mean().values
    df_sloping['avg_score'] = group['predict'].mean().values

    plt.figure(figsize=(12, 3), dpi=200)
    plt.subplot(1, 4, 1)
    plt.plot(fpr, tpr, linewidth=0.8)
    plt.plot((0, 1), (0, 1), color='k', linestyle='dashed', linewidth=0.5)
    plt.plot(fpr[diff.argmax()], tpr[diff.argmax()], 'r.', markersize=5)
    plt.axis(xmin=0.0, xmax=1.0)
    plt.axis(ymin=0.0, ymax=1.0)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('false positive rate', fontsize=6)
    plt.ylabel('true positive rate', fontsize=6)
    plt.title('AUC = {0}'.format(round(auc_value, 3)), fontsize=7)

    plt.subplot(1, 4, 2)
    plt.hist(predict, bins=30, normed=True, facecolor='mediumaquamarine', alpha=0.9)
    plt.axvline(x=np.mean(predict), color='powderblue', linestyle='dashed', linewidth=0.7)
    plt.axvline(x=np.mean(truth), color='lightcoral', linestyle='dashed', linewidth=0.7)
    plt.title('Tru = {0},  Pred = {1}'.format(round(truth.mean(), 3), round(predict.mean(), 3)), fontsize=7)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('score', fontsize=6)
    plt.ylabel('probability', fontsize=6)

    plt.subplot(1, 4, 3)
    plt.plot(np.linspace(0, 1, diff.size), tpr, linewidth=0.8, color='cornflowerblue', label='TPR')
    plt.plot(np.linspace(0, 1, diff.size), fpr, linewidth=0.8, color='firebrick', label='FPR')
    plt.plot(np.linspace(0, 1, diff.size), diff, linewidth=0.8, color='slategray', label='TPR - FPR')
    plt.plot((maxidx, maxidx), (0.0, ks), linewidth=0.4, color='r')
    plt.axis(xmin=0.0, xmax=1.0)
    plt.axis(ymin=0.0, ymax=1.0)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.ylabel('tpr / fpr', fontsize=6)
    plt.legend(loc=2, fontsize=6)
    plt.title('KS = {0}, Thres = {1}, Reject {2}%'.format(round(ks, 3), round(cut_point, 4), reject_porp), fontsize=7)

    plt.subplot(1, 4, 4)
    plt.plot(df_sloping['bin'], df_sloping['avg_score'], 'b.-', linewidth=0.8, label='Prediction', markersize=3)
    plt.plot(df_sloping['bin'], df_sloping['target_rate'], 'r.-', linewidth=0.8, label='Truth', markersize=3)
    plt.axhline(predict.mean(), color='powderblue', linestyle='dashed', linewidth=0.7, label='Overall Avg score')
    plt.axhline(truth.mean(), color='lightcoral', linestyle='dashed', linewidth=0.7, label='Overall Target rate')
    plt.legend(loc=2, fontsize=6)
    plt.xticks(df_sloping['bin'], fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('bin', fontsize=6)
    plt.ylabel('target rate', fontsize=6)
    plt.title('Sample = {0}, Bins = {1}'.format(n, df_sloping.shape[0]), fontsize=7)

    if title is not None:
        plt.suptitle(title, fontsize=10, x=0.02, y=1.04, horizontalalignment='left')
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'model_performance({0}).png'.format(title)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    plt.show()
    plt.close()

    return df_sloping, auc_value, ks


def calc_listing_overdue_rate(df, score, target, bins=10, plot=False, title='', save_path=None):
    """
    根据score等样本量分bin，计算每个bin中的累积标的逾期率
    Parameters
    ----------
    df: DataFrame
    score: str
        分数列名
    target: str
        target列名
    bins: int or list, default 10
        分bin的方式，如果是int则表示将模型分从小到大排列后均匀分成几个bin，如果是list则按指定切分点分bin
    plot: bool, default False
        是否绘图
    title: str, default None
        图片标题
    save_path: str, default None
        图片存储路径

    Returns
    -------
    result: DataFrame
        每个bin中的累积逾期率
    """
    df_result = pd.DataFrame()
    df_tmp = df[[score, target]].copy()
    if type(bins) == int:
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='equal')
    elif type(bins) in (list, np.ndarray):
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='quantile')
    else:
        raise ValueError('bins type can only be [int, list, np.ndarray]')
    group = df_tmp.groupby('bin')

    sample_cnt = group[target].count()
    overdue_cnt = group[target].sum()
    df_result['bin'] = np.arange(1, len(sample_cnt) + 1)
    df_result['odue_rate'] = (overdue_cnt / sample_cnt).values
    df_result['cum_odue_rate'] = (overdue_cnt.cumsum() / sample_cnt.cumsum()).values

    if plot:
        plt.figure(figsize=(4, 3), dpi=120)
        plt.plot(df_result['bin'], df_result['odue_rate'], color='red', linewidth=0.6, marker='.', markersize=2,
                 label='overdue rate')
        plt.plot(df_result['bin'], df_result['cum_odue_rate'], color='lightcoral', linewidth=0.6, marker='.',
                 markersize=2, label='cumulative overdue rate')
        plt.xticks(df_result['bin'], fontsize=5)
        plt.yticks(fontsize=5)
        plt.xlabel('bin', fontsize=5)
        plt.ylabel('target rate', fontsize=5)
        plt.legend(loc=2, fontsize=5)
        plt.title('Overdue rate{0}'.format(title), fontsize=7)

        if save_path is not None:
            if save_path.endswith('.png') or save_path.endswith('.jpg'):
                plt.savefig(save_path, bbox_inches='tight')
            elif os.path.isdir(save_path):
                plt.savefig(os.path.join(save_path, 'target_rate_plot.png'), bbox_inches='tight')
            else:
                raise ValueError('No such file or directory: {0}'.format(save_path))
        plt.show()
        plt.close()

    return df_result


def calc_amount_overdue_rate(df, score, principal, dueamount, bins=10, plot=False, title='', save_path=None):
    """
    根据score等样本量分bin，计算每个bin中的金额逾期率
    Parameters
    ----------
    df: DataFrame
    score: str
        分数列名
    principal: str
        借款本金列名
    dueamount: str
        逾期本金列名
    bins: int or list, default 10
        分bin的方式，如果是int则表示将模型分从小到大排列后均匀分成几个bin，如果是list则按指定切分点分bin
    plot: bool, default False
        是否绘图
    title: str, default None
        图片标题
    save_path: str, default None
        图片存储路径

    Returns
    -------
    result: DataFrame
        每个bin中的累积逾期率
    """
    df_result = pd.DataFrame()
    df_tmp = df[[score, principal, dueamount]].copy()
    if type(bins) == int:
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='equal')
    elif type(bins) in (list, np.ndarray):
        df_tmp['bin'] = cut_bins(df_tmp[score], bins=bins, method='quantile')
    else:
        raise ValueError('bins type can only be [int, list, np.ndarray]')
    group = df_tmp.groupby('bin')
    total_principal = group[principal].sum()
    total_dueamount = group[dueamount].sum()
    df_result['bin'] = np.arange(1, len(total_principal) + 1)
    df_result['amt_odue_rate'] = (total_dueamount / total_principal).values
    df_result['cum_amt_odue_rate'] = (total_dueamount.cumsum() / total_principal.cumsum()).values

    if plot:
        plt.figure(figsize=(4, 3), dpi=120)
        plt.plot(df_result['bin'], df_result['amt_odue_rate'], color='blue', linewidth=0.6, marker='.', markersize=2,
                 label='amount overdue rate')
        plt.plot(df_result['bin'], df_result['cum_amt_odue_rate'], color='cornflowerblue', linewidth=0.6, marker='.',
                 markersize=2, label='cumulative amount overdue rate')
        plt.xticks(df_result['bin'], fontsize=5)
        plt.yticks(fontsize=5)
        plt.xlabel('bin', fontsize=5)
        plt.ylabel('overdue rate', fontsize=5)
        plt.legend(loc=2, fontsize=5)
        plt.title('Amount overdue rate{0}'.format(title), fontsize=7)

    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'amount_overdue_rate.png'), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    plt.show()
    plt.close()

    return df_result


def bad_sample_rank_scatter(df, score, target, cutoff=None, title=None, save_path=None):
    """
    绘制坏样本模型分排序的散点图
    df: DataFrame
    score: str
        分数的列名
    target: str
        target列名
    cutoff: list, default None
        切分点，切分的百分比, eg. [0.05, 0.2, 0.5, 0.8, 0.9]
    title: str, default None
        图片标题
    save_path: str, default None
        图片存储路径
    """
    score_rank = df[score].rank(method='first').astype(int)  # 模型分排序
    # good_sample_rank = score_rank.loc[df[target] == 0].values
    bad_sample_rank = score_rank.loc[df[target] == 1].values

    plt.figure(figsize=(16, 5), dpi=400)
    #     plt.scatter(x=good_sample_rank, y=np.repeat(1, good_sample_rank.shape[0]),
    #                 s=np.repeat(0.1, good_sample_rank.shape[0]), c='w', label='good')
    plt.scatter(x=bad_sample_rank, y=np.repeat(1, bad_sample_rank.shape[0]),
                s=np.repeat(0.4, bad_sample_rank.shape[0]), c='r', label='bad sample')

    # 加垂直分割线
    if cutoff is not None:
        cut_rank = score_rank.quantile(cutoff)
        for p in cut_rank:
            plt.axvline(p, color='gray', linestyle='dashed', linewidth=0.6)
    plt.legend(loc=1, fontsize=12)
    plt.axis(xmin=0.0, xmax=score_rank.max())
    plt.axis(ymin=0.0, ymax=2)
    plt.yticks([], fontsize=8)
    plt.xlabel('rank', fontsize=10)
    if title is not None:
        plt.title(title, fontsize=15)

    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, 'bad_sample_scatter.png'), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    plt.show()
    plt.close()


def meanofbucket(df, x, score, target, type='numerical', bins=10, to_show=True, save_path=None):
    """
    根据变量x分bin，计算每个bin中样本score的均值和target rate
    Parameters
    ----------
    df: DataFrame
    x: str
        用于分bin的变量名
    score: str
        score列名
    target: str
        target列名
    type: str, default 'numerical' options ['numerical', 'categorical']
        变量类型
    bins: int or list, default 10
        分bin的方式，如果是int则表示将变量x从小到大排列后均匀分成几个bin，如果是list则按指定分bin方式切分
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存储路径

    Returns
    -------
    result: DataFrame
        每个bin中变量的均值
    """
    df_tmp = df[[x, score, target]].copy()
    if type == 'numerical':
        if isinstance(bins, int):
            df_tmp['bin'], cut_points = cut_bins(df_tmp[x], bins=bins, method='quantile')
        else:
            df_tmp['bin'] = cut_bins(df_tmp[x], bins=bins, method='point')
            cut_points = bins
    elif type == 'categorical':
        df_tmp['bin'] = df_tmp[x]
    else:
        raise ValueError('type can only be "numerical" or "categorical"]')

    group = df_tmp.groupby('bin')
    result = group[x].count().to_frame().reset_index(drop=False)
    result.columns = ['bin', 'sample_count']
    # result['bin_lb'] = cut_points
    result['proportion'] = 1.0 * result['sample_count'] / df_tmp.shape[0]
    # result['bin'] = result['bin'].astype(int)
    result['prediction'] = group[score].mean().values
    result['truth'] = group[target].mean().values
    result['residual'] = np.abs(result['truth'] - result['prediction'])

    x_coor = np.arange(result.shape[0])
    if type == 'numerical':
        shift = np.arange(result.shape[0]) + 0.5
        xticks = [round(i, 4) for i in cut_points]
    elif type == 'categorical':
        shift = np.arange(result.shape[0])
        xticks = result['bin']

    plt.figure(figsize=(8, 3), dpi=200)
    plt.subplot(1, 2, 1)
    plt.plot(shift, result['truth'], color='firebrick', linewidth=0.7, marker='.', markersize=2,
             label='truth')
    plt.plot(shift, result['prediction'], color='cornflowerblue', linewidth=0.7, marker='.', markersize=2,
             label='prediction')
    plt.plot(shift, result['residual'], color='mediumaquamarine', linestyle='dashed', linewidth=0.7,
             label='residual')
    plt.axis(ymin=0.0, ymax=result[['prediction', 'truth']].max().max() * 1.5)
    plt.xticks(x_coor, xticks, rotation=60, fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel(x, fontsize=6)
    plt.ylabel('mean', fontsize=6)
    plt.legend(loc=2, fontsize=6)
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.4)
    plt.title('Mean of prediction and truth in each bucket', fontsize=7)

    plt.subplot(1, 2, 2)
    plt.bar(shift, result['proportion'], color='lightgray', align='center', width=0.95)
    plt.xticks(x_coor, xticks, rotation=60, fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel(x, fontsize=6)
    plt.ylabel('proportion', fontsize=6)
    ax = plt.gca()
    for at in ['left', 'right', 'bottom', 'top']:
        ax.spines[at].set_linewidth(0.4)
    plt.title('Sample Proportion in each bucket', fontsize=7)

    plt.suptitle(x, fontsize=10, x=0.5, y=1.01)
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '{0}.png'.format(x)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()

    return result


#
# def swap_in_out_anlys(df_score, score_new, score_old, amount, target, ind_deal, buffer=1.0, bin_num=10):
#     header = ['passrate', 'overdue_rt_diff', 'overdue_rt_diff_rto', 'amount_overdue_rt_diff',
#               'amount_overdue_rt_diff_rto']
#     df_result = pd.DataFrame(columns=header)
#
#     # 计算切分点并分bin
#     cut_points_new = df_score[score_new].quantile(np.arange(0.0, 1.0, 1 / bin_num))
#     cut_points_old = df_score[score_old].quantile(np.arange(0.0, 1.0, 1 / bin_num))
#     cut_points_new = np.append(cut_points_new, 1.0)
#     cut_points_old = np.append(cut_points_old, 1.0)
#     df_score['bin_new'] = pd.cut(df_score[score_new], bins=cut_points_new, precision=13, right=True,
#                                  include_lowest=True, labels=range(1, bin_num + 1))
#     df_score['bin_old'] = pd.cut(df_score[score_old], bins=cut_points_old, precision=13, right=True,
#                                  include_lowest=True, labels=range(1, bin_num + 1))
#     df_score_deal = df_score[df_score[ind_deal] == 1].reset_index(drop=True)
#     df_score_overdue = df_score[df_score[target] == 1].reset_index(drop=True)
#
#     # 计算交叉矩阵
#     cnt_mat = pd.crosstab(df_score['bin_new'], df_score['bin_old']).fillna(0)  # 新老模型每个bin的人数的交叉矩阵
#     overdue_rt_mat = df_score_deal.pivot_table(index='bin_new', columns='bin_old', values=target,
#                                                aggfunc='mean', fill_value=0.0)  # 新老模型每个bin的逾期率
#     amount_mat = df_score.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                       aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的发标金额
#     deal_amount_mat = df_score_deal.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                                 aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的发标金额
#     overdue_amount_mat = df_score_overdue.pivot_table(index='bin_new', columns='bin_old', values=amount,
#                                                       aggfunc='sum', fill_value=0.0)  # 新老模型每个bin的逾期金额
#     amount_overdue_rt_mat = (overdue_amount_mat / deal_amount_mat).fillna(0.0)
#
#     for i, passrate in enumerate(np.arange(0.0 + 1.0 / bin_num, 1.0, 1.0 / bin_num)):
#         overdue_rt_new = (cnt_mat.iloc[0:i + 1, :] * overdue_rt_mat.iloc[0:i + 1, :]).sum().sum() / cnt_mat.iloc[
#                                                                                                     0:i + 1,
#                                                                                                     :].sum().sum()
#         overdue_rt_old = (cnt_mat.iloc[:, 0:i + 1] * overdue_rt_mat.iloc[:, 0:i + 1]).sum().sum() / cnt_mat.iloc[:,
#                                                                                                     0:i + 1].sum().sum()
#         amount_overdue_rt_new = (amount_mat.iloc[0:i + 1, :] * amount_overdue_rt_mat.iloc[0:i + 1,
#                                                                :]).sum().sum() / amount_mat.iloc[0:i + 1, :].sum().sum()
#         amount_overdue_rt_old = (amount_mat.iloc[:, 0:i + 1] * amount_overdue_rt_mat.iloc[:,
#                                                                0:i + 1]).sum().sum() / amount_mat.iloc[:,
#                                                                                        0:i + 1].sum().sum()
#
#         df_result.loc[i, 'passrate'] = passrate
#         df_result.loc[i, 'overdue_rt_diff'] = overdue_rt_old - overdue_rt_new
#         df_result.loc[i, 'overdue_rt_diff_rto'] = (overdue_rt_old - overdue_rt_new) / overdue_rt_old
#         df_result.loc[i, 'amount_overdue_rt_diff'] = amount_overdue_rt_old - amount_overdue_rt_new
#         df_result.loc[i, 'amount_overdue_rt_diff_rto'] = (
#                                                          amount_overdue_rt_old - amount_overdue_rt_new) / amount_overdue_rt_old
#
#     return df_result
#
#
# def shift_matrix(df_base, df_shift, var_name, join_key):
#     """
#     计算转移矩阵
#     """
#     if type(join_key) == str:
#         var_list = [join_key] + [var_name]
#     elif type(join_key) == list:
#         var_list = join_key + [var_name]
#
#     df_merge = pd.merge(df_base[var_list], df_shift[var_list], how='inner', on=join_key, suffixes=['_base', '_shift'])
#
#     if df_merge.shape[0] != df_base.shape[0]:
#         print('base数据与shift数据的scope不完全重合')
#     mat = pd.crosstab(df_merge[var_name + '_base'], df_merge[var_name + '_shift'], margins=True, normalize='index')
#
#     return mat
