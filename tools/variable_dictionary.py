# coding: utf-8
#!/root/anaconda/bin/python


__all__ = ['sql2dict', 'create_dictionary']


import os
from datetime import datetime
import re
from pandas import DataFrame


def __sql_request(file_path):
    """
    读取sql文件生成字符串
    """
    sql_file = open(file_path)
    try:
        sql = sql_file.read()
    except:
        raise Exception
    finally:
        sql_file.close()

    return sql


def __gettablename(sql):
    """
    提取建表语句的表名
    """
    table_view_pattern = re.compile(r'create +(table|view) +([^ ]+)', re.IGNORECASE)
    table_name = table_view_pattern.findall(sql)
    
    return table_name


def __getsourcetable(sql):
    """
    提取建表语句中使用到的表名
    """
    source_pattern = re.compile(r'(?:from|join(?:\s*\[shuffle\])?)\s+([^ ()\s]+)\s*', re.IGNORECASE)
    source_table = source_pattern.findall(sql)

    return source_table


def __getorigintable(table_name, source_table, res=None):
    """
    提取建表语句的底层依赖表
    """
    if res is None:
        res = []
    if table_name not in source_table.keys():
        res.append(table_name)
        return
    
    for table in source_table[table_name]:
        __getorigintable(table, source_table, res)
    
    return set(res)


def sql2dict(sql, dict_col=['No.', 'Var name', 'Meaning']):
    """
    根据sql代码字符串生成数据字典
    Parameters
    ----------
    sql: str
        SQL代码字符串
    dict_col: list, default ['No.', 'Var name', 'Meaning']
        数据字典的字段名，需要和sql代码中的变量注释一一对应，默认为No.(变量编号)、Var name(变量名)、Comment(变量注释)
        
    Returns
    -------
    var_dict: DataFrame
        数据字典
    """
    var_dict = DataFrame(columns=dict_col + ['Source table'])
    sql_list = []
    source_table = {}

    for s in sql.split(';'):
        s = s.lower()
        sql_list.append(s)

    for sql_part in sql_list:
        if sql_part.find('create table') > -1:
            new_table = __gettablename(sql_part)[0][1]
            source_table[new_table] = __getsourcetable(sql_part)

    i = 0
    for sql_part in sql_list:
        if sql_part.find('create') > -1 and sql_part.find('@var:') > -1:
            from_table = __gettablename(sql_part)[0][1]
            origin_table = __getorigintable(from_table, source_table=source_table)
            var_list = re.findall(r"@var:.*@", sql_part)
            for var in var_list:
                line = var.replace('@var:', '').replace('@', '').lower().split(',')
                if len(line) != len(dict_col):
                    raise Exception('This variable commit is error : {0}'.format(var))
                line.append(', '.join(origin_table))
                var_dict.loc[i, :] = line
                i += 1

    return var_dict


def create_dictionary(sql_file, dict_col=['No.', 'Var name', 'Comment'], save_path=None):
    """
    根据.sql文件或文件路径批量生产数据字典，并保存为带有时间戳的csv文件
    Parameters
    ----------
    sql_file: str
        sql文件路径或文件夹路径，若为文件夹路径则根据该路径下所有.sql文件生成数据字典
    dict_col: list, default ['No.', 'Var name', 'Meaning']
        数据字典的字段名，需要和sql代码中的变量注释一一对应，默认为No.(变量编号)、Var name(变量名)、Comment(变量注释)
    save_path: str
        数据字典存放路径

    Returns
    -------
    var_dict: DataFrame
    """
    if os.path.isfile(sql_file):
        sqls = __sql_request(sql_file)
        var_dict = sql2dict(sqls, dict_col=dict_col)
    elif os.path.isdir(sql_file):
        sqls = ''
        for file_name in os.listdir(sql_file):
            if file_name.endswith('.sql'):
                sqls += '\n' + __sql_request(os.path.join(sql_file, file_name))
        var_dict = sql2dict(sqls, dict_col=dict_col)
    else:
        raise Exception('请输入正确的SQL文件名或路径')

    # 检查是否有字段重复
    for col in ['Var name', 'Comment']:
        value_cnt = var_dict[col].value_counts()
        if (value_cnt > 1).sum() > 0:
            print("{0}存在重复值: {1}".format(col, list(value_cnt.index[value_cnt > 1])))

    if save_path is not None:
        if save_path.endswith('.csv'):
            var_dict.to_csv(save_path, index=False, encoding='utf-8')
        elif save_path.endswith('.xlsx'):
            var_dict.to_excel(save_path, index=False)
        elif os.path.isdir(save_path):
            var_dict.to_excel(os.path.join(save_path, 'variable_dictionary_{0}.csv'.format(str(datetime.now())[:19])), index=False)
        else:
            raise ValueError('No such file or directory: {0}'.format(save_path))

    return var_dict


if __name__ == '__main__':
    create_dictionary(os.getcwd(), dict_col=['No.', 'Var name', 'Comment', 'Author'], save_path=os.getcwd())
