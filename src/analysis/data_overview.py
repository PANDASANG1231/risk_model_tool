# coding: utf-8
# !/root/anaconda/bin/python
# Author: Jingcheng Qiu


import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


usage = """
此工具包包含计算变量的分布和稳定性指标
"""


def edd(df, missing_value=np.nan, ignore_col=[], save_path=None):
    """
    计算变量的一些常规统计指标，其中数值型的均值，分位数等是去掉-1之后统计的
    Parameters
    ----------
    df: DataFrame
    missing_value: int、float、str， default np.nan
        变量缺失值
    ignore_col: list, default []
        不进行计算的变量列表
    save_path: str
        结果文件存储路径

    Returns
    -------
    edd_result: DataFrame
    """
    start_time = datetime.now()
    header = ['var', 'type', 'sample_cnt', 'unique_values', 'missingrate', 'zerorate', 'mode', 'mean', 'std', 'min',
              'qt1', 'qt5', 'qt25', 'qt50', 'qt75', 'qt95', 'qt99', 'max']
    edd_result = pd.DataFrame(columns=header)
    sample_cnt = df.shape[0]

    for i, var in enumerate(df.columns):
        if var in ignore_col:
            continue
        try:
            var_type = type(df[var][df[var].isnull() == 0].iloc[0])
        except:
            var_type = type(df[var].iloc[0])
        edd_result.loc[i, 'var'] = var
        edd_result.loc[i, 'type'] = var_type.__name__

        if var_type == str:
            if missing_value in [np.nan, np.NaN]:
                nomissingdata = df.loc[~((df[var].isnull()) | (df[var] == '')), var]
            else:
                nomissingdata = df.loc[df[var] != missing_value, var]
            edd_result.loc[i, 'unique_values'] = nomissingdata.unique().size
            edd_result.loc[i, 'missingrate'] = 1.0 * (sample_cnt - nomissingdata.shape[0]) / sample_cnt
            if nomissingdata.shape[0] > 0:
                edd_result.loc[i, 'mode'] = nomissingdata.mode().values[0]

        elif var_type.__name__.startswith('int') or var_type.__name__.startswith('float'):
            if missing_value in [np.nan, np.NaN]:
                nomissingdata = df.loc[~df[var].isnull(), var]
            else:
                nomissingdata = df.loc[df[var] != missing_value, var]
            quantiles = nomissingdata.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).values
            edd_result.loc[i, 'unique_values'] = nomissingdata.unique().size
            edd_result.loc[i, 'missingrate'] = 1.0 * (sample_cnt - nomissingdata.shape[0]) / sample_cnt
            edd_result.loc[i, 'zerorate'] = (nomissingdata == 0.0).sum() / sample_cnt
            edd_result.loc[i, 'mean'] = nomissingdata.mean()
            edd_result.loc[i, 'std'] = nomissingdata.std()
            edd_result.loc[i, 'min'] = quantiles[0]
            edd_result.loc[i, 'qt1'] = quantiles[1]
            edd_result.loc[i, 'qt5'] = quantiles[2]
            edd_result.loc[i, 'qt25'] = quantiles[3]
            edd_result.loc[i, 'qt50'] = quantiles[4]
            edd_result.loc[i, 'qt75'] = quantiles[5]
            edd_result.loc[i, 'qt95'] = quantiles[6]
            edd_result.loc[i, 'qt99'] = quantiles[7]
            edd_result.loc[i, 'max'] = quantiles[8]

        else:
            raise Exception('unknown data type: {0}'.format(var))
    edd_result['sample_cnt'] = sample_cnt

    if save_path is not None:
        if save_path.endswith('.csv'):
            edd_result.to_csv(save_path, index=False, encoding='utf-8')
        elif save_path.endswith('.xlsx'):
            edd_result.to_excel(save_path, index=False)
        elif os.path.isdir(save_path):
            edd_result.to_csv(os.path.join(save_path, 'edd_result.csv'), index=False, encoding='utf-8')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    print('\n====================== time cost: {} =====================\n'.format(datetime.now() - start_time))

    return edd_result


def var_distribution(datasets, var, quantiles=np.arange(0, 1, 0.1), names=None, colors=None, to_show=True, save_path=None):
    """
    分组绘制变量变量的分布图
    Parameters
    ----------
    datasets: list
        DataFrame的序列
    var: str
        变量名
    names: list, default None
        数据集的名字
    quantiles: np.array, default np.arange(0, 1, 0.1)
        分位点
    colors: list, default None
        每个数据集绘图的颜色
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存储路径

    Returns
    -------
    edd_result: DataFrame
    """
    if names is None:
        names = ['dataset%s' % i for i in range(1, len(datasets) + 1)]
    if len(datasets) != len(names):
        raise ValueError('datasets and names must have same length')
    header = ['var_name', 'dataset', 'missingrate', 'zerorate', 'mean', 'quantiles']
    df_result = pd.DataFrame(columns=header)

    for i, df in enumerate(datasets):
        df_result.loc[i, 'var_name'] = var
        df_result.loc[i, 'dataset'] = names[i]
        df_result.loc[i, 'missingrate'] = df[var].isnull().mean()
        df_result.loc[i, 'zerorate'] = (df[var] == 0.0).mean()
        df_result.loc[i, 'mean'] = df[var].mean()
        df_result.loc[i, 'quantiles'] = df[var].quantile(quantiles).values

    plt.subplots(1, 2, figsize=(7, 3), dpi=200)
    if colors is None:
        colors = plt.cm.Paired(np.linspace(0.1, 1, len(names)))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(names)), 1.0, width=.4, color='lightgray',  label='Normal Value')
    plt.bar(np.arange(len(names)), df_result['missingrate'] + df_result['zerorate'], width=.4, color='slategray', label='Zero Value')
    plt.bar(np.arange(len(names)), df_result['missingrate'], width=.4, color='firebrick', label='Missing Value')
    plt.axis(ymin=0, ymax=1.2)
    plt.xticks(range(len(names)), names, fontsize=6)
    plt.yticks(np.arange(0, 1.2, 0.2), ['0', '20%', '40%', '60%', '80%', '100%'], fontsize=6)
    plt.xlabel('group', fontsize=6)
    plt.ylabel('percent', fontsize=6)
    plt.legend(loc=2, fontsize=5)
    plt.title('Missing & Zero value rate', fontsize=8)

    plt.subplot(1, 2, 2)
    for i in range(len(names)):
        plt.plot(np.arange(quantiles.size), df_result.loc[i, 'quantiles'], color=colors[i], linewidth=0.7, label=names[i])
    xticks = [str(round(i, 2)) for i in quantiles]
    plt.xticks(np.arange(quantiles.size), xticks, rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel('quantile', fontsize=6)
    plt.legend(loc=2, fontsize=5)
    plt.title('Variable Quantile Values', fontsize=8)

    plt.suptitle(var, fontsize=10, x=0.5, y=1.01)
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '{0}.png'.format(var)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()

    return df_result


def vlm(df, var, date, period='month'):
    """
    根据周期看变量的趋势，计算指标有缺失率、均值、标准差、中位数、25%分位点、75%分位点
    Parameters
    ----------
    df: DataFrame
    var: str
        变量名
    date: str
        日期名
    period: str, default 'month', options ['day', 'week', 'month', 'year']
        图片存储路径

    Returns
    -------
    df_vlm: DataFrame
        变量不同时期样本的统计指标
    population_index: dict
        全量样本的统计指标
    """
    try:
        datetime.strptime(df[date].iloc[0], '%Y-%m-%d')
    except:
        raise Exception('"{0}" is not a correct date format'.format(date))
    df_tmp = df[[var, date]].copy()

    if period == 'day':
        df_tmp['period'] = df_tmp[date].apply(lambda x: x[0:10])
    elif period == 'week':
        df_tmp['period'] = '9999-12-31'
        week_start_date = df[date].min()
        week_end_date = week_start_date
        max_date = df[date].max()
        while week_end_date < max_date:
            week_end_date = datetime.strftime(datetime.strptime(week_start_date, '%Y-%m-%d') + timedelta(7), '%Y-%m-%d')
            df_tmp.loc[(df_tmp[date] >= week_start_date) & (df_tmp[date] < week_end_date), 'period'] = week_start_date
            week_start_date = week_end_date
    elif period == 'month':
        df_tmp['period'] = df_tmp[date].apply(lambda x: x[0:7])
    elif period == 'year':
        df_tmp['period'] = df_tmp[date].apply(lambda x: x[0:4])
    else:
        raise ValueError('period can only be "day", "week", "month" or "year"')

    group = df_tmp.groupby('period')
    df_vlm = group['period'].count().to_frame()
    df_vlm.columns = ['sample_cnt']
    df_vlm['missingcnt'] = group[var].apply(lambda x: x.isnull().sum()).to_frame()
    df_vlm['missingrate'] = 1.0 * df_vlm['missingcnt'] / df_vlm['sample_cnt']
    df_vlm['zerocnt'] = group[var].apply(lambda x: (x == 0.0).sum())
    df_vlm['zerorate'] = 1.0 * df_vlm['zerocnt'] / df_vlm['sample_cnt']
    df_vlm['mean'] = group[var].mean()
    df_vlm['median'] = group[var].median()
    df_vlm['std'] = group[var].std()
    df_vlm['qt25'] = group[var].quantile(0.25)
    df_vlm['qt75'] = group[var].quantile(0.75)
    df_vlm['var_name'] = var
    df_vlm = df_vlm.reset_index(drop=False)

    # 计算全量样本的统计指标
    population_index = dict()
    population_index['missingrate'] = 1.0 * df_tmp[var].isnull().sum() / df_tmp.shape[0]
    population_index['zerorate'] = 1.0 * (df_tmp[var] == 0.0).sum() / df_tmp.shape[0]
    population_index['mean'] = df_tmp[var].mean()
    population_index['median'] = df_tmp[var].median()
    population_index['std'] = df_tmp[var].std()
    population_index['qt25'] = df_tmp[var].quantile(0.25)
    population_index['qt75'] = df_tmp[var].quantile(0.75)

    return df_vlm, population_index


def vlm_plot(df_vlm, population_index, elasticity=0.3, to_show=True, save_path=None):
    """
    绘制变量的vlm图像
    Parameters
    ----------
    df_vlm: DataFrame
    population_index: dict
        全量样本的统计量
    elasticity: float, default 0.3
        上界和下界的弹性
    to_show: bool, default True
        是否展示图片
    save_path: str, default None
        图片存储路径
    """
    xticks_list = df_vlm['period'].tolist()
    rows = len(xticks_list)
    var_name = df_vlm['var_name'].iloc[0]

    plt.subplots(3, 2, figsize=(12, 12), dpi=200)
    plt.subplot(3, 2, 1)
    plt.bar(range(rows), df_vlm['sample_cnt'], width=.3, color='lightgray',  label='Sample count')
    plt.bar(range(rows), df_vlm['missingcnt'] + df_vlm['zerocnt'], width=.3, color='slategray', label='Zero count')
    plt.bar(range(rows), df_vlm['missingcnt'], width=.3, color='firebrick', label='Missing count')
    plt.axis(ymin=0, ymax=1.4 * df_vlm['sample_cnt'].max())
    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.ylabel('sample count', fontsize=8)
    plt.legend(loc=2, fontsize=7)
    plt.twinx()
    plt.plot(range(rows), df_vlm['zerorate'], color='slategray', label='Zero rate')
    plt.plot(range(rows), df_vlm['missingrate'], color='firebrick', label='Missing rate')
    plt.axhline(population_index['zerorate'], color='lightsteelblue', linestyle='dashed',
                linewidth=0.8, label='Overall Zero rate')
    plt.axhline(population_index['missingrate'], color='lightcoral', linestyle='dashed',
                linewidth=0.8, label='Overall Missing rate')
    plt.yticks(np.arange(0.0, 1.3, 0.1), ['0', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.yticks(fontsize=6)
    plt.ylabel('value rate', fontsize=8)
    plt.legend(loc=1, fontsize=7)
    plt.title('Missing & Zero value rate', fontsize=11)

    plt.subplot(3, 2, 2)
    plt.plot(range(rows), df_vlm['mean'], color='firebrick', label='mean')
    plt.axhline((1 + elasticity) * population_index['mean'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Upper boundary')
    plt.axhline((1 - elasticity) * population_index['mean'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Lower boundary')
    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.axis(ymin=0, ymax=1.7 * df_vlm['mean'].max())
    plt.legend(loc=2, fontsize=8)
    plt.title('Mean', fontsize=11)

    plt.subplot(3, 2, 3)
    plt.plot(range(rows), df_vlm['median'], color='firebrick', label='median')
    plt.axhline((1 + elasticity) * population_index['median'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.7, label='Upper boundary')
    plt.axhline((1 - elasticity) * population_index['median'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.7, label='Lower boundary')
    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.axis(ymin=0, ymax=1.7 * df_vlm['median'].max())
    plt.legend(loc=2, fontsize=8)
    plt.title('Median', fontsize=11)

    plt.subplot(3, 2, 4)
    plt.plot(range(rows), df_vlm['std'], color='firebrick', label='std')
    plt.axhline((1 + elasticity) * population_index['std'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Upper boundary')
    plt.axhline((1 - elasticity) * population_index['std'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Lower boundary')
    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.axis(ymin=0, ymax=1.7 * df_vlm['std'].max())
    plt.legend(loc=2, fontsize=8)
    plt.title('Standard deviation', fontsize=11)

    plt.subplot(3, 2, 5)
    plt.plot(range(rows), df_vlm['qt25'], color='firebrick', label='qt25')
    plt.axhline((1 + elasticity) * population_index['qt25'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Upper boundary')
    plt.axhline((1 - elasticity) * population_index['qt25'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Lower boundary')

    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.axis(ymin=0, ymax=1.7 * df_vlm['qt25'].max())
    plt.legend(loc=2, fontsize=8)
    plt.title('25% Quantile', fontsize=11)

    plt.subplot(3, 2, 6)
    plt.plot(range(rows), df_vlm['qt75'], color='firebrick', label='qt75')
    plt.axhline((1 + elasticity) * population_index['qt75'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Upper boundary')
    plt.axhline((1 - elasticity) * population_index['qt75'], color='cornflowerblue', linestyle='dashed',
                linewidth=0.8, label='Lower boundary')
    plt.xticks(range(rows), xticks_list, rotation=60, fontsize=8)
    plt.yticks(fontsize=6)
    plt.axis(ymin=0, ymax=1.7 * df_vlm['qt75'].max())
    plt.legend(loc=2, fontsize=8)
    plt.title('75% Quantile', fontsize=11)

    plt.suptitle(var_name, fontsize=15, x=0.5, y=1.01)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    if save_path is not None:
        if save_path.endswith('.png') or save_path.endswith('.jpg'):
            plt.savefig(save_path, bbox_inches='tight')
        elif os.path.isdir(save_path):
            plt.savefig(os.path.join(save_path, '{0}.png'.format(var_name)), bbox_inches='tight')
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))
    if to_show:
        plt.show()
    plt.close()
