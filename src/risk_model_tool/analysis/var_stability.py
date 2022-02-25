# coding: utf-8
# Author: Jingcheng Qiu


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from plot_chs_config import zhfont
from woe_tools import woe_calc, iv_calc


def calc_psi(delp, valid):
    """
    计算模型的PSI指标
    Args:
        delp: array, 新数据每个bin中样本的占比
        valid: array, 建模数据每个bin中样本的占比，默认为0.1
    Return:
        psi: Float
    """
    if len(delp) != len(valid):
        raise ValueError('Different shape of arguments!')
    psi = np.sum((delp - valid) * np.log(delp / valid))

    return psi


def create_monitor_file(monitor_name):
    """
    创建监控文件模板
    """
    if monitor_name == 'var_psi':
        header = ['dataset', 'var_name', 'psi']

    elif monitor_name == 'var_woe_tend':
        header = ['dataset', 'var_name', 'bin_no', 'var_value', 'ref_value', 'new_ref_value', 'count_0',
                  'count_1', 'total', 'target_rate', 'proportion', 'iv_adj']

    df = pd.DataFrame(columns=header)

    return df


class Stability(object):
    def __init__(self, var_dict, woe_ref, target, part_values):
        self.var_dict = var_dict
        self.woe_ref = woe_ref
        self.target = target
        self.part_values = part_values
        self.psi = None
        self.woe_tend = None

    def var_psi_monitor(self, datasets):
        """
        计算变量的PSI，按WOE时的分bin区间分bin，对应文件var_psi.csv
        Parameters
        ----------
        datasets: list
            Woe后的数据集
        """
        df_psi = create_monitor_file('var_psi')

        n = 0
        for i, part in enumerate(self.part_values):
            df_part = datasets[i]
            for var in self.var_dict['numerical'] + self.var_dict['categorical']:
                if var in self.var_dict['numerical']:
                    woe_var = 'nwoe_' + var
                else:
                    woe_var = 'cwoe_' + var
                # 根据WOE reference table计算训练集上的取值占比
                train_prop = self.woe_ref.loc[self.woe_ref['Var_Name'] == var, ['Ref_Value', 'Proportion']]
                # 计算新的数据的取值占比
                delp_value_prop = df_part[woe_var].value_counts() / df_part.shape[0]
                delp_prop = pd.DataFrame(delp_value_prop).reset_index(drop=False)
                delp_prop.columns = ['Ref_Value', 'Delp_Proportion']
                df_prop = pd.merge(train_prop, delp_prop, how='left', on='Ref_Value').dropna()
                psi = calc_psi(df_prop['Proportion'].values, df_prop['Delp_Proportion'].values)
                df_psi.loc[n, 'dataset'] = part
                df_psi.loc[n, 'var_name'] = var
                df_psi.loc[n, 'psi'] = psi
                n += 1

        self.psi = df_psi

    def var_woe_tend_monitor(self, datasets):
        """
        分组监控变量的WOE值趋势变化，对应文件var_woe_tend.csv
        Parameters
        ----------
        datasets: list
            Woe后的数据集
        """
        df_woe_tend = create_monitor_file('var_woe_tend')

        for var in self.var_dict['numerical'] + self.var_dict['categorical']:
            if var in self.var_dict['numerical']:
                woe_var = 'nwoe_' + var
            else:
                woe_var = 'cwoe_' + var
            left_table = self.woe_ref.loc[
                self.woe_ref['Var_Name'] == var, ['Var_Name', 'Bin_No', 'Var_Value', 'Ref_Value', 'IV']].reset_index(
                drop=True)
            left_table.columns = ['var_name', 'bin_no', 'var_value', 'ref_value', 'iv']
            left_table['ref_value'] = left_table['ref_value'].apply(lambda x: round(x, 5))

            var_res = []  # 用于存储每部分数据集计算的结果
            for i, part in enumerate(self.part_values):
                df_part = datasets[i]
                # 重新计算每个bin的woe值
                groups = df_part.groupby(woe_var)
                ds = groups[self.target].value_counts().unstack().sort_index().fillna(0)
                ds['ratio_0'] = ds[0] / ds[0].sum()
                ds['ratio_1'] = ds[1] / ds[1].sum()
                ds['total'] = ds[0] + ds[1]
                ds['proportion'] = ds['total'] / df_part.shape[0]
                ds['target_rate'] = ds[1] / ds['total']
                ds['new_ref_value'] = ds.apply(lambda x: woe_calc(x[1], x[0], ds[1].sum(), ds[0].sum()), axis=1)
                ds['new_iv'] = iv_calc(ds[1], ds[0])
                ds = ds.reset_index(drop=False)
                ds.columns = ['ref_value', 'count_0', 'count_1', 'ratio_0', 'ratio_1', 'total', 'proportion',
                              'target_rate', 'new_ref_value', 'new_iv']

                # 计算IV的惩罚项
                ds['penalty'] = np.abs(ds['ratio_1'] - ds['ratio_0']) * np.abs(ds['new_ref_value'] - ds['ref_value'])
                ds['ref_value'] = ds['ref_value'].apply(lambda x: round(x, 5))  # 取woe值5位小数,防止精度问题merge不上
                ds = pd.merge(left_table, ds, how='left', on=['ref_value'])
                ds['dataset'] = part
                var_res.append(ds)

            woe_tend_var = pd.concat(var_res, axis=0, ignore_index=True)
            woe_tend_var['iv_adj'] = woe_tend_var['new_iv'].mean() - woe_tend_var['penalty'].sum() / len(
                self.part_values)
            df_woe_tend = pd.concat([df_woe_tend, woe_tend_var], axis=0, ignore_index=True)

        df_woe_tend = df_woe_tend[['dataset', 'var_name', 'bin_no', 'var_value', 'count_0', 'count_1', 'ratio_0', 'ratio_1',
                                   'total', 'target_rate', 'proportion', 'ref_value', 'new_ref_value', 'penalty', 'iv',
                                   'new_iv', 'iv_adj']]
        self.woe_tend = df_woe_tend

    def plot_stability_graph(self, to_show=True, save_path=None):
        """
        绘制变量稳定性图
        Parameters
        ----------
        to_show: bool, default True
            是否展示图片
        save_path: str, default None
            图片存储路径
        """
        colors = plt.cm.Paired(np.linspace(0.1, 1, len(self.part_values)))

        for var in self.var_dict['numerical'] + self.var_dict['categorical']:
            print(var)
            df_psi_var = self.psi.loc[self.psi['var_name'] == var, :].reset_index(drop=True)
            df_woe_tend_var = self.woe_tend.loc[self.woe_tend['var_name'] == var, :].reset_index(drop=True)
            df_proportion_pivot = df_woe_tend_var.pivot_table(index='dataset', columns='bin_no', values='proportion')
            df_woe_value_pivot = df_woe_tend_var.pivot_table(index='dataset', columns='bin_no', values='new_ref_value')
            bins = df_woe_tend_var['bin_no'].unique()

            plt.subplots(1, 2, figsize=(12, 5), dpi=200)
            # PSI图
            plt.subplot(1, 2, 1)
            plt.plot(np.arange(len(self.part_values)), df_psi_var['psi'], color='mediumpurple')
            plt.xticks(np.arange(len(self.part_values)), self.part_values, fontsize=8)
            plt.ylabel('PSI', fontsize=8)
            plt.ylabel('Dataset', fontsize=8)
            plt.title('Average PSI = {0}'.format(round(df_psi_var['psi'].mean(), 4)), fontsize=12)

            # WOE趋势图
            plt.subplot(1, 2, 2)
            for i, part in enumerate(self.part_values):
                plt.bar(x=bins + 0.1 * (i - len(self.part_values) + 1), height=df_proportion_pivot.loc[part, :],
                        width=0.1, color=colors[i], label=part)
            plt.axis(ymin=0, ymax=1.0)
            plt.xticks(bins, bins, fontsize=8)
            plt.ylabel('Proportion', fontsize=8)
            plt.xlabel('Bin number', fontsize=8)
            plt.legend(loc=2, fontsize=6)
            plt.twinx()
            for i, part in enumerate(self.part_values):
                plt.plot(bins, df_woe_value_pivot.loc[part, :], color=colors[i], label=part)
            plt.ylabel('Woe value', fontsize=8)
            plt.legend(loc=1, fontsize=6)
            plt.title('Adjust IV = {0}'.format(round(df_woe_tend_var['iv_adj'].iloc[0], 5)), fontsize=12)
            plt.suptitle(var, fontsize=16, x=0.5, y=1.04, horizontalalignment='center')
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
