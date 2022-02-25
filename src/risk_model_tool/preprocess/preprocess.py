# coding: utf-8
# Author: Jingcheng Qiu

import os
import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np
import woe_tools as woe


usage = '''
################################### Summarize #######################################
此工具包用于数据预处理，包含以下内容：
1.Cap
2.Floor
3.MissingImpute
4.Woe
5.Normalize
6.Scale
7.Tactic
-------------------------------------------------------------------------------------
使用说明：
import pandas as pd
import numpy as np
import preprocess as pp

df_train = pd.read_csv('train_data.csv')
df_test = pd.read_csv('test_data.csv')
df_config = pd.read_csv('edd_config.csv')

# 调用单个组件:
operation = pp.MissingImpute(df_config)
df_reference = operation.fit(df_train)
df_train = operation.apply(df_train)
df_test = operation.apply(df_test)

# 设计整个数据预处理流程:
process = pp.Tactic(df_config, process_list=[pp.Cap, pp.Floor, pp.MissingImpute, pp.Woe], target='target')
process.summary()
df_reference = process.fit(df_train)
df_train = process.apply(df_train)
df_test = process.apply(df_test)
process.save_reference('./edd_reference.csv')

# 也可以通过读入一个已经生成的reference table，直接对数据进行apply处理
df_reference = pd.read_csv('edd_reference.csv')
process = pp.Tactic(df_reference, process_list=[pp.Cap, pp.Floor, pp.MissingImpute, pp.Woe], target='target')
df_train = process.apply(df_train)
df_test = process.apply(df_test)
---------------------------------------------------------------------------------------
注意事项：
1. 不要在数据存在缺失值的情况下进行woe处理；
2. 当处理流程中包含woe时，必须指定target，否则会报错；
3. 对于一个新的数据集，第一次做处理时最好分步进行预处理，方便检查每步的输出是否正确。
#######################################################################################
'''


def __map_feature_type(t, time_as_num=False):
    """
    convert the dataFrame type to feature type (Numerical or Categorical)
    """
    if t in (int, np.int64, np.int32, np.int16, bool, float, np.float32, np.float64, np.float128):
        return 'numerical'

    elif t in (str,):
        return 'categorical'

    elif t in (pd.tslib.Timestamp, ):
        return 'numerical' if time_as_num else 'timestamp'


def __extract_feature_type(df, known_columns={}):
    """
    extract columns type of a dataframe and map it
    """
    col_list = []
    for var in df.columns:
        if var in known_columns:
            col_list.append((var, known_columns[var]))
            continue

        var_type = __map_feature_type(df[var].dtype.type)
        if var_type is not None:
            col_list.append((var, var_type))
            continue

        type_set = set(df[var][~df[var].isnull()].apply(lambda x: type(x)))
        if len(type_set) == 1:
            var_type = __map_feature_type(type_set.pop())
            if var_type is not None:
                col_list.append((var, var_type))
                continue
        raise ValueError('Unknown type of column "{0}" as {1}'.format(var, type_set))

    return col_list


def create_edd_config(df_master, known_columns={}, save_path=None):
    """
    生成数据预处理的config文件
    Parameters
    ----------
    df_master:
        DataFrame
    known_columns: dict, default {}
        已知的列类型，eg. {'age': 'numerical, 'sex': 'categorical'}
    save_path: str, default None

    Returns
    -------
    df_config: DataFrame
        预处理的配置文件
    """
    column_type = __extract_feature_type(df_master, known_columns=known_columns)
    df_config = pd.DataFrame(column_type, columns=['Var_Name', 'Var_Type'])
    df_config['Ind_Model'] = 1    # 是否进模型
    df_config['Ind_Cap'] = 0    # 是否进行Cap处理
    df_config['Cap_Value'] = None
    df_config['Ind_Floor'] = 0    # 是否进行Floor处理
    df_config['Floor_Value'] = None
    df_config['Missing_Impute'] = -1    # 填入的缺失值，数值型变量默认为-1，字符变量默认为'missing'
    df_config.loc[df_config['Var_Type'] == 'categorical', 'Missing_Impute'] = 'missing'
    df_config['Ind_WOE'] = 0    # 是否做WOE变换，默认数值型变量不做变换，字符型变量做
    df_config.loc[df_config['Var_Type'] == 'categorical', 'Ind_WOE'] = 1
    df_config['WOE_Bin'] = None
    df_config['Ind_Norm'] = 0   # 是否进行normalize
    df_config['Ind_Scale'] = 0  # 是否进行min-max scale

    for var in df_config['Var_Name'][df_config['Var_Type'] == 'numerical'].tolist():
        if df_master[var].max() > (5 * df_master[var].quantile(0.99)):
            df_config.loc[df_config['Var_Name'] == var, 'Ind_Cap'] = 1

    df_config.to_csv(save_path, index=False, encoding='utf-8')

    return df_config


class Cap(object):
    """
    Descriptions
    ------------
    对变量做cap处理，主要包括以下几点：
    1. 只对numerical的变量做处理
    2. cap操作默认用5倍p99(有指定值优先用指定值)
    3. 对missing值不处理

    Atributes
    ---------
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的cap值
    apply: 根据reference table对变量做cap处理
    """
    def __init__(self, df_config, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件(必填)
        apply_list: list, default None
            需要处理的变量列表，若未指定，则为config文件中Ind_Cap=1的变量
        """
        self.config = df_config
        self.reference = df_config.copy()
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Ind_Cap'] == 1)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master):
        """
        计算变量的cap值
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        for var in self.apply_list:
            df_config_var = self.config[self.config['Var_Name'] == var]
            if df_config_var['Cap_Value'].isnull().iloc[0] == True:
                cap_value = df_master[var][~df_master[var].isnull()].quantile(0.99)    # 忽略缺失值
            else:
                cap_value = float(df_config_var['Cap_Value'].iloc[0])
            self.reference.loc[self.reference['Var_Name'] == var, 'Cap_Value'] = cap_value

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量做cap处理
        Parameters
        ----------
        df_master: DataFrame
        """
        for var in self.apply_list:
            cap_value = float(self.reference['Cap_Value'][self.reference['Var_Name'] == var].iloc[0])
            if pd.isnull(cap_value):
                raise ValueError('Not found cap value of "{0}"'.format(var))
            df_master[var] = np.where(df_master[var] > cap_value, cap_value, df_master[var])

        return df_master


class Floor(object):
    """
    Descriptions
    ------------
    对变量做floor处理，主要包括以下几点：
    1. 只对numerical的变量做处理
    2. 只对小于0的值做处理，默认用5p1(有指定值优先用指定值)
    3. 对missing值不处理

    Attributes
    ---------
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的floor值
    apply: 根据reference table对变量做floor处理
    """
    def __init__(self, df_config, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件
        apply_list: list, default None
            需要处理的变量列表，若未指定，则为config中Ind_Floor=1的变量
        """
        self.config = df_config
        self.reference = df_config.copy()
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Ind_Floor'] == 1)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master):
        """
        计算变量的floor值
        Parameters
        ----------
            df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        for var in self.apply_list:
            df_config_var = self.config[self.config['Var_Name'] == var]
            if df_config_var['Floor_Value'].isnull().iloc[0] == True:
                floor_value = min(5 * df_master[var][~df_master[var].isnull()].quantile(0.01), 0)
            else:
                floor_value = float(df_config_var['Floor_Value'].iloc[0])
            self.reference.loc[self.reference['Var_Name'] == var, 'Floor_Value'] = floor_value

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量做floor处理
        Parameters
        ----------
        df_master: DataFrame
        """
        for var in self.apply_list:
            floor_value = float(self.reference['Floor_Value'][self.reference['Var_Name'] == var].iloc[0])
            if pd.isnull(floor_value):
                raise ValueError('Not found floor value of "{0}"'.format(var))
                df_master[var] = np.where(df_master[var] < floor_value, floor_value, df_master[var])

        return df_master


class MissingImpute(object):
    """
    Descriptions
    ------------
    对变量进行缺失值填充，主要包括以下几点：
    1. 对于numerical变量有mean/median/指定值三种填充方式
    2. 对于categorical变量有mode/指定值两种填充方式
    3. 某个变量存在缺失值但没有指定填充值时会给出警告

    Attributes
    ---------
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的填充值
    apply: 根据reference table对变量进行缺失值填充
    """
    def __init__(self, df_config, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件
        apply_list: list, default None
            需要处理的变量列表，若未指定，则为config文件中Missing_Impute不为空的变量
        """
        self.config = df_config
        self.reference = df_config.copy()
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Missing_Impute'].isnull() == False)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master):
        """
        计算变量的填充值
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        missing_cnt = df_master.isnull().sum()  # 统计各变量的缺失值数量
        missing_vars = list(missing_cnt[missing_cnt > 0].index)  # 筛选存在缺失值的变量

        for var in list(self.config['Var_Name'][self.config['Ind_Model'] == 1]):
            df_config_var = self.config[self.config['Var_Name'] == var]

            # 确定numerical变量的填充值
            if df_config_var['Var_Type'].iloc[0] == 'numerical':
                if df_config_var['Missing_Impute'].iloc[0] == 'mean':
                    impute_value = df_master[var].mean()
                elif df_config_var['Missing_Impute'].iloc[0] == 'median':
                    impute_value = df_master[var].median()
                elif df_config_var['Missing_Impute'].isnull().iloc[0] == False:
                    impute_value = float(df_config_var['Missing_Impute'].iloc[0])
                else:
                    impute_value = None

            # 确定categorical变量的填充值
            elif df_config_var['Var_Type'].iloc[0] == 'categorical':
                if df_config_var['Missing_Impute'].iloc[0] == 'mode':
                    impute_value = df_master[var].mode().iloc[0]
                elif df_config_var['Missing_Impute'].isnull().iloc[0] == False:
                    impute_value = df_config_var['Missing_Impute'].iloc[0]
                else:
                    impute_value = None

            # 未知的变量类型报错
            else:
                raise TypeError('Wrong type for:{0}'.format(var))

            # 更新config文件
            self.reference.loc[self.reference['Var_Name'] == var, 'Missing_Impute'] = impute_value

            # 检查存在缺失值但未指定填充值的变量
            if var in list(self.config['Var_Name'][self.config['Ind_Model'] == 1]) and var in missing_vars:
                if impute_value is None:
                    print('"{0}" exist missing value but no impute!'.format(var))

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量进行缺失值填充
        Parameters
        ----------
        df_master: DataFrame
        """
        missing_cnt = df_master.isnull().sum()
        missing_vars = list(missing_cnt[missing_cnt > 0].index)

        for var in self.apply_list:
            if var not in missing_vars:
                continue
            if self.reference['Var_Type'][self.reference['Var_Name'] == var].iloc[0] == 'numerical':
                impute_value = float(self.reference['Missing_Impute'][self.reference['Var_Name'] == var].iloc[0])
            else:
                impute_value = self.reference['Missing_Impute'][self.reference['Var_Name'] == var].iloc[0]
            if pd.isnull(impute_value):
                raise ValueError('Not found impute value of "{0}"'.format(var))
            df_master[var] = df_master[var].fillna(impute_value)

        return df_master


class Woe(object):
    """
    Descriptions
    ------------
    对变量做woe处理，主要包括以下几点:
    1. 分numerical和categorical两类变量处理
    2. 默认采用自动分bin形式，如果有自填bin，优先使用
    3. save_path默认为"./woe"，也可在fit时指定

    Attributes
    ----------
    config: DataFrame
        config table
    target: str
        target变量名
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的woe和iv值
    apply: 根据reference table对变量进行woe替换，并删除woe前的原始变量
    """
    def __init__(self, df_config, target, woe_ref=None, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFram
            数据预处理的config文件
        target: str
            target变量的名字
        woe_ref: DataFrame, default None
            woe reference table
        apply_list: list, default None
            需要处理的变量列表，若未指定，则为config文件中Ind_WOE=1的变量
        """
        self.config = df_config
        self.reference = woe_ref
        self.target = target
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Ind_WOE'] == 1)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master, batch_save=0, to_plot=True, save_path=os.getcwd()+'/woe_result'):
        """
        计算变量的woe和iv值
        Parameters
        ----------
        df_master: DataFrame
        batch_save: int
            分批储存，每隔多少个变量存一次reference文件，默认为0，即不分批存储
        to_plot: bool, default True
            是否绘图
        save_path: str
            woe图和reference table的输出路径，默认为'./woe_result'

        Returns
        -------
        reference: DataFrame
            woe reference table
        """
        df_woe_config = self.config[[var in self.apply_list for var in self.config['Var_Name']]]
        df_ref_num = pd.DataFrame()
        df_ref_cag = pd.DataFrame()
        num_vars_list = list(df_woe_config['Var_Name'][df_woe_config['Var_Type'] == 'numerical'])
        cag_vars_list = list(df_woe_config['Var_Name'][df_woe_config['Var_Type'] == 'categorical'])

        # 创建输出文件夹
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        elif os.path.isdir(save_path) == os.getcwd():
            os.mkdir(os.path.join(save_path, 'woe_result'))

        # 将WOE_Bin转化成list
        dict_woe_bin = df_woe_config[['Var_Name', 'WOE_Bin']][(df_woe_config['Var_Type'] == 'numerical') & ((df_woe_config['WOE_Bin'].isnull() == False))].set_index('Var_Name').to_dict()
        num_dict_woe_bin = {var: [float(i.strip('[] ')) for i in dict_woe_bin['WOE_Bin'][var].split(',')] for var in dict_woe_bin['WOE_Bin']}

        # 计算变量的WOE值
        cnt = 1
        for var in num_vars_list:
            print('\n WOE process for: {0}'.format(var))
            if var in num_dict_woe_bin.keys():
                df_ref_num_tmp = woe.numwoe_aptbinning(df_master, var_name=var, bins=num_dict_woe_bin[var],
                                                       target=self.target, to_plot=to_plot, save_path=save_path)
            else:
                df_ref_num_tmp = woe.numwoe_autobinning(df_master, var_name=var, target=self.target, max_bins=6,
                                                        missing=True, to_plot=to_plot, save_path=save_path)
            df_ref_num = pd.concat((df_ref_num, df_ref_num_tmp), axis=0)
            if batch_save > 0 and cnt % batch_save == 0:
                df_ref_num.to_csv(os.path.join(save_path, 'num_woe_ref_table.csv'), index=False, encoding='utf-8')
            cnt += 1

        for var in cag_vars_list:
            print('\n WOE process for : {0}'.format(var))
            df_master[var] = df_master[var].apply(lambda x: woe.str_convert(x))
            df_ref_cag_tmp = woe.catwoe_autobinning(df_master, var_name=var, target=self.target, max_bins=6,
                                                    missing_value='-1', to_plot=to_plot, save_path=save_path)
            df_ref_cag = pd.concat((df_ref_cag, df_ref_cag_tmp), axis=0)
            if batch_save > 0 and cnt % batch_save == 0:
                df_ref_cag.to_csv(os.path.join(save_path, 'cag_woe_ref_table.csv'), index=False, encoding='utf-8')
            cnt += 1

        df_woe_ref = pd.concat((df_ref_num, df_ref_cag), axis=0)
        df_woe_ref.to_csv(os.path.join(save_path, 'woe_ref_table.csv'), index=False, encoding='utf-8')
        self.reference = df_woe_ref

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量进行woe替换，并删除woe前的原始变量
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        df_return: DataFrame
        """
        if self.reference is None:
            raise ValueError('Reference table missing!')

        df_return = df_master.copy()
        df_woe_ref = self.reference[[var in self.apply_list for var in self.reference['Var_Name']]]
        df_ref_num = df_woe_ref[['Var_Name', 'Var_Value', 'Ref_Value']][df_woe_ref['Var_Type'] == 'numerical'].copy()
        df_ref_cag = df_woe_ref[['Var_Name', 'Var_Value', 'Ref_Value']][df_woe_ref['Var_Type'] == 'categorical'].copy()

        # numerical变量woe替换
        if df_ref_num.shape[0] > 0:
            for var in df_ref_num['Var_Name'].unique():
                df_ref_var = df_ref_num[df_ref_num['Var_Name'] == var]
                woe.numwoe_apply(df_return, ref_table=df_ref_var, var_name=var)

        # categorical变量woe替换
        if df_ref_cag.shape[0] > 0:
            for var in df_ref_cag['Var_Name'].unique():
                df_ref_var = df_ref_cag[df_ref_cag['Var_Name'] == var]
                woe.catwoe_apply(df_return, ref_table=df_ref_var, var_name=var)

        keep_list = [var for var in df_return.columns if var not in self.apply_list]   # 删除经过WOE之后的原始变量

        return df_return[keep_list]


class Normalize(object):
    """
    Descriptions
    ------------
    对变量进行标准化

    Attributes
    ----------
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的mean和standard deviation
    apply: 根据reference table对变量进行标准化
    """
    def __init__(self, df_config, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件
        apply_list: list
            需要处理的变量列表，若未指定，则为config文件中Ind_Norm=1的变量
        """
        self.config = df_config
        self.reference = df_config.copy()
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Ind_Norm'] == 1)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master):
        """
        计算变量的mean和standard deviation
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        mean = []
        std = []
        df_ref = pd.DataFrame(self.apply_list, columns=['Var_Name'])
        for var in self.apply_list:
            var_type = self.config['Var_Type'][self.config['Var_Name'] == var].iloc[0]
            # 防止config文件中没有Ind_WOE列导致报错
            try:
                ind_woe = self.config['Ind_WOE'][self.config['Var_Name'] == var].iloc[0]
            except:
                ind_woe = 0
            if ind_woe == 1:
                if var_type == 'numerical':
                    real_var = 'nwoe_' + var
                elif var_type == 'categorical':
                    real_var = 'cwoe_' + var
            else:
                real_var = var
            var_mean = df_master[real_var].mean()
            var_std = df_master[real_var].std()
            mean.append(var_mean)
            std.append(var_std)
        df_ref['Var_Name'] = self.apply_list
        df_ref['Mean'] = mean
        df_ref['Std'] = std
        self.reference = pd.merge(self.config, df_ref, how='left', on='Var_Name')

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量进行标准化
        Parameters
        ----------
        df_master: DataFrame
        """
        for var in self.apply_list:
            var_type = self.config['Var_Type'][self.config['Var_Name'] == var].iloc[0]
            try:
                ind_woe = self.config['Ind_WOE'][self.config['Var_Name'] == var].iloc[0]
            except:
                ind_woe = 0
            if ind_woe == 1:
                if var_type == 'numerical':
                    real_var = 'nwoe_' + var
                elif var_type == 'categorical':
                    real_var = 'cwoe_' + var
            else:
                real_var = var
            var_mean = self.reference['Mean'][self.reference['Var_Name'] == var].iloc[0]
            var_std = self.reference['Std'][self.reference['Var_Name'] == var].iloc[0]
            if pd.isnull(var_mean) or pd.isnull(var_std):
                raise ValueError('Not found mean or std of "{0}"'.format(var))
            if var_std == 0.0:
                raise ValueError('Standard deviation equal 0: {0}'.format(var))
            df_master[real_var] = (df_master[real_var] - var_mean) / var_std

        return df_master


class Scale(object):
    """
    Descriptions
    ------------
    对变量进行Min-Max变换，并对大于1或小于0的值做cap-floor处理。

    Atributes:
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    apply_list: list
        需要处理的变量列表

    Method
    ------
    fit: 计算变量的min和max
    apply: 根据reference table对变量进行标准化
    """
    def __init__(self, df_config, apply_list=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件
        apply_list: list, default None
            需要处理的变量列表，若未指定，则为config文件中Ind_Scale=1的变量
        """
        self.config = df_config
        self.reference = df_config.copy()
        if apply_list is None:
            self.apply_list = list(df_config['Var_Name'][(df_config['Ind_Model'] == 1) & (df_config['Ind_Scale'] == 1)])
        else:
            self.apply_list = apply_list

    def fit(self, df_master):
        """
        计算变量的min和max
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        min = []
        max = []
        df_ref = pd.DataFrame(self.apply_list, columns=['Var_Name'])
        for var in self.apply_list:
            var_type = self.config['Var_Type'][self.config['Var_Name'] == var].iloc[0]
            # 防止config文件中没有Ind_WOE列导致报错
            try:
                ind_woe = self.config['Ind_WOE'][self.config['Var_Name'] == var].iloc[0]
            except:
                ind_woe = 0
            if ind_woe == 1:
                if var_type == 'numerical':
                    real_var = 'nwoe_' + var
                elif var_type == 'categorical':
                    real_var = 'cwoe_' + var
            else:
                real_var = var
            var_min = df_master[real_var].min()
            var_max = df_master[real_var].max()
            min.append(var_min)
            max.append(var_max)
        df_ref['Min'] = min
        df_ref['Max'] = max
        self.reference = pd.merge(self.config, df_ref, how='left', on='Var_Name')

        return self.reference

    def apply(self, df_master):
        """
        根据reference table对变量进行min-max transform
        Parameters
        ----------
        df_master: DataFrame
        """
        for var in self.apply_list:
            var_type = self.config['Var_Type'][self.config['Var_Name'] == var].iloc[0]
            try:
                ind_woe = self.config['Ind_WOE'][self.config['Var_Name'] == var].iloc[0]
            except:
                ind_woe = 0
            if ind_woe == 1:
                if var_type == 'numerical':
                    real_var = 'nwoe_' + var
                elif var_type == 'categorical':
                    real_var = 'cwoe_' + var
            else:
                real_var = var
            var_min = self.reference['Min'][self.reference['Var_Name'] == var].iloc[0]
            var_max = self.reference['Max'][self.reference['Var_Name'] == var].iloc[0]
            if pd.isnull(var_min) or pd.isnull(var_max):
                raise ValueError('Not found max or min of "{0}", please check your config file!'.format(var))
            if (var_max - var_min) == 0.0:
                raise ValueError('Max equal min: {0}'.format(var))
            df_master[real_var] = (df_master[real_var] - var_min) / (var_max - var_min)
            df_master.loc[df_master[real_var] > 1.0, real_var] = 1.0
            df_master.loc[df_master[real_var] < 0.0, real_var] = 0.0

        return df_master


class Tactic(object):
    """
    Descriptions
    ------------
    按顺序对数据进行多种处理，可选用Cap、Floor、MissingImpute、Woe、Normlize, Scale六种处理组件

    Atributes
    ---------
    config: DataFrame
        config table
    reference: DataFrame
        reference table
    process_list: list
        预处理组件的列表
    target: str
        target变量名
    woe_reference: DataFrame
        woe reference table

    Method
    ------
    check_config_file: 检查配置文件是否正确
    add_process: 添加处理组件
    clear_process_list: 清空当前的process_list
    add_variables: 将变量的Ind_Model变更为1
    drop_variables: 将变量的Ind_Model变更为0
    summary: 打印当前预处理的流程
    fit: 计算变量的min和max
    apply: 根据reference table对变量进行标准化
    save_reference: 保存fit后的reference table
    """
    def __init__(self, df_config, process_list=None, target=None, woe_ref=None):
        """
        Parameters
        ----------
        df_config: DataFrame
            数据预处理的config文件
        process_list: list, default None
            预处理组件的列表，eg. [Cap, Floor, Missingimpute, Woe]
        target: str, default None
            target变量名, 若不做WOE可不填
        woe_ref: DataFrame, default None
            woe reference table，若不做WOE可不填
        """
        self.config = df_config
        self.reference = df_config.copy()
        if process_list is None:
            self.process_list = []
        else:
            self.process_list = process_list
        self.target = target
        self.woe_reference = woe_ref

    def __check_attribute(self):
        """
        检查当要做WOE时，target是否填写
        """
        if Woe in self.process_list and self.target is None:
            raise ValueError('Target is must need when use Woe process!')

    def __complie(self, operation):
        """
        编译处理流程
        """
        if operation.__name__ == 'Woe':
            instance = operation(self.reference, woe_ref=self.woe_reference, target=self.target)
        else:
            instance = operation(self.reference)

        return instance

    def check_config_file(self, df_master):
        """
        检查config文件是否正确，主要有以下几点：
        1. 检查config文件的header名是否正确；
        2. 检查config里的变量是否都在df_master中；
        3. 检查Var_Type是否正确填写；
        4. 检查是否有categorical变量做cap & floor处理；
        5. 检查Impute_Value是否合理。

        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        cnt_error: int
            config文件中的错误个数
        """
        cnt_error = 0
        # 检查header是否正确
        allow_header = ['Var_Name', 'Var_Type', 'Preicesion', 'Ind_Model', 'Ind_Cap', 'Cap_Value', 'Ind_Floor',
                        'Floor_Value', 'Missing_Impute', 'Ind_WOE', 'WOE_Bin', 'Ind_Norm', 'Ind_Scale']
        config_header = list(self.config.columns)
        wrong_header = [col for col in config_header if col not in allow_header]
        if wrong_header:
            print('Wrong header: {0}'.format(wrong_header))
            cnt_error += 1

        # 检查Var_Name是否正确
        var_list_config = list(self.config['Var_Name'])
        var_list_master = list(df_master.columns)
        wrong_var_name = [var for var in var_list_config if var not in var_list_master]
        if wrong_var_name:
            print('These vars is not in master table: {0}'.format(wrong_var_name))
            cnt_error += 1

        # 检查Var_Type是否正确
        var_type_master = df_master.dtypes
        num_var_list = list(self.config['Var_Name'][self.config['Var_Type'] == 'numerical'])
        cag_var_list = list(self.config['Var_Name'][self.config['Var_Type'] == 'categorical'])
        num_type_list = [int, float, np.int64, np.int32, np.int16, np.int8, np.float64, np.float32]
        cag_type_list = [str, np.object]
        for var in var_list_config:
            if var not in num_var_list and var not in cag_var_list:
                print("Wrong variable type, only allow use numerical or categorical: {0}").format(var)
                cnt_error += 1
            elif var in num_var_list and var_type_master[var] not in num_type_list:
                print('Type of {0} is {1}, but write numerical'.format(var, var_type_master[var]))
                cnt_error += 1
            elif var in cag_var_list and var_type_master[var] not in cag_type_list:
                print('Type of {0} is {1}, but write categorical'.format(var, var_type_master[var]))
                cnt_error += 1

        # 检查是否有categorical变量做cap & floor处理
        for var in cag_var_list:
            if self.config['Ind_Cap'][self.config['Var_Name'] == var].iloc[0] == 1:
                print('{0} is categorical variable, but use cap process'.format(var))
                cnt_error += 1
            elif self.config['Ind_Floor'][self.config['Var_Name'] == var].iloc[0] == 1:
                print('{0} is categorical variable, but use floor process'.format(var))
                cnt_error += 1

        # 检查Impute_Value是否合理
        for var in num_var_list:
            impute_value = self.config['Missing_Impute'][self.config['Var_Name'] == var]
            if type(impute_value) == str and impute_value not in ['mean', 'median']:
                print('Wrong impute value, only allow use mean/median: var:{0} value:{1}'.format(var, impute_value))
                cnt_error += 1

        return cnt_error

    def add_process(self, operation, target=None, woe_ref=None):
        """
        添加预处理组件
        Parameters
        ----------
        operation: object or list
            组件名或列表
        target: str, default None
            target变量名，当使用Woe处理时必须填入
        woe_ref: DataFrame, default None
            woe reference table，当直接使用Woe apply处理时必须填入
        """
        if type(operation) != list:
            operation = list(operation)
        for op in operation:
            if op not in self.process_list:
                self.process_list.append(op)
                if op.__name__ == 'Woe':
                    if self.woe_reference is None:
                        self.woe_reference = woe_ref
                    if self.target is None:
                        self.target = target
            else:
                print('"{0}" has existed'.format(op.__name__))

    def clear_process_list(self):
        """
        清空process_list
        """
        self.process_list = []

    def add_variables(self, add_list, column='Ind_Model'):
        """
        添加进入模型的变量
        Parameters
        ----------
        add_list: list
            要加入模型的变量名列表
        column: str, default 'Ind_Model'
            往什么流程中加入变量
        """
        if type(add_list) == str:
            add_list = [add_list]
        for var in add_list:
            self.reference.loc[self.reference['Var_Name'] == var, column] = 1
        print('加入{0}个变量到{1}'.format(len(add_list), column))

    def drop_variables(self, drop_list, column='Ind_Model'):
        """
        删除进入模型的变量
        Parameters
        ----------
        drop_list: list
            从模型中移除的变量名列表
        column: str, default 'Ind_Model'
            从什么流程中删除变量
        """
        if type(drop_list) == str:
            drop_list = [drop_list]
        for var in drop_list:
            self.reference.loc[self.reference['Var_Name'] == var, column] = 0
        print('从{0}中删除{1}个变量'.format(column, len(drop_list)))

    def summary(self):
        """
        打印当前预处理流程信息
        """
        operation_list = [operation.__name__ for operation in self.process_list]
        print(' -----> '.join(operation_list))

    def fit(self, df_master):
        """
        根据数据拟合预处理的值
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        reference: DataFrame
            reference table
        """
        df_tmp = df_master.copy()
        error_cnt = self.check_config_file(df_master)    # 检查config文件
        if error_cnt > 0:
            raise TypeError('Must pass config check before fit!')
        self.__check_attribute()     # 检查attribute是否完备
        for operation in self.process_list:
            start_time = datetime.now()
            print('============================= {0} Fitting ============================='.format(operation.__name__))
            instance = self.__complie(operation=operation)
            if operation.__name__ == 'Woe':
                self.woe_reference = instance.fit(df_tmp, batch_save=1)
            else:
                self.reference = instance.fit(df_tmp)
            df_tmp = instance.apply(df_tmp)
            print('Time cost: {0}'.format(datetime.now() - start_time))
        return self.reference

    def apply(self, df_master):
        """
        根据reference table对数据进行预处理
        Parameters
        ----------
        df_master: DataFrame

        Returns
        -------
        df_return: DataFrame
        """
        df_return = df_master
        for operation in self.process_list:
            start_time = datetime.now()
            print('============================= {0} Applying ============================='.format(operation.__name__))
            instance = self.__complie(operation=operation)
            df_return = instance.apply(df_return)
            print('Time cost: {0}'.format(datetime.now() - start_time))

        return df_return

    def save_reference(self, save_path='reference_table.csv'):
        """
        保存reference table
        Parameters
        ----------
        save_path: str
            文件的保存路径，默认为'reference_table.csv'
        """
        self.reference.to_csv(save_path, index=False, encoding='utf-8')
