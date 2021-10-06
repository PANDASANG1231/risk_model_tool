# coding: utf-8


from datetime import datetime
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import numpy as np
from scipy import sparse


def onehot_matrix(applist, colnums, save_sparse=True):
    """
    根据appid，生成app的one-hot稀疏矩阵
    Parametes
    ---------
    applist: list or array
        每个样本的appid list, eg.[[5, 74, 28], [32, 48, 3]]
    colnums: int
        one-hot矩阵的列数
    save_sparse: bool, default True
        是否存为稀疏矩阵

    Returns
    -------
    mat: sparse.csr_matrix or array
        one-hot矩阵
    """
    rownums = applist.size
    indices = np.empty(0, dtype=int)  # 列索引
    indptr = np.zeros(rownums + 1, dtype=int)  # 非零元素索引
    for i, row in enumerate(applist):
        indices = np.append(indices, row)
        indptr[i + 1] = indptr[i] + len(row)

    values = np.ones(indptr[-1], dtype=float)
    mat = sparse.csr_matrix((values, indices, indptr), shape=(rownums, colnums), dtype=float)
    if save_sparse is False:
        mat = mat.toarray()

    return mat


def batch_onehot(applist, colnums, batch_size=1, save_sparse=True, verbose=-1):
    """
    分批生成applist one-hot稀疏矩阵
    Parametes
    ---------
    applist: list or array
        每个样本的appid list, eg.[[1, 15, 28], [32, 48, 3]]
    colnums: int
        one-hot矩阵的列数
    batch_size: int, default 1
        分批数
    save_sparse: bool, default True
        是否存为稀疏矩阵
    verbose: int, default 1
        每隔多少批显示一次进度, 若设置成-1则不显示当前进度

    Returns
    -------
    mat: sparse.csr_matrix or array
        one-hot矩阵
    """
    if isinstance(batch_size, int) and batch_size > 0:
        idx = np.linspace(0, applist.size, batch_size + 1, dtype=int)  # 生成切片索引序列
    else:
        raise ValueError('batch_size must be a positive integer')

    if isinstance(applist, pd.Series):
        applist = applist.values

    # batch process
    stack = []
    start_time = datetime.now()
    print('========================== Start ==========================')
    for b in range(idx.size - 1):
        if verbose > 0 and (b + 1) % verbose == 0:
            print('========================== Batch: {0}/{1} =========================='.format(b + 1, batch_size))
        indices = applist[idx[b]:idx[b + 1]]
        stack.append(onehot_matrix(indices, colnums=colnums))

    mat = sparse.vstack(stack)
    if save_sparse is False:
        mat = mat.toarray()
    print('========================== Time cost: {0} =========================='.format(datetime.now() - start_time))

    return mat


# def batch_multi_onehot(rowdata, colnums, batch_size=1, n_jobs=1, save_sparse=True):
#     """
#     多线程分批生成applist one-hot稀疏矩阵
#     Parametes
#     ---------
#     rowdata: list or array
#         每个样本的appid list
#     colnums: int
#         one-hot矩阵的列数
#     batch_size: int, default 1
#         分批数
#     n_jobs: int, default 1
#         进程数
#     save_sparse: bool, default True
#         是否存为稀疏矩阵

#     Returns
#     -------
#     mat: sparse.csr_matrix or array
#         one-hot矩阵
#     """
#     if isinstance(batch_size, int) and batch_size > 0:
#         idx = np.linspace(0, rowdata.size, batch_size + 1, dtype=int)  # 生成切片索引序列
#     else:
#         raise ValueError('batch_size must be a positive integer')

#     if n_jobs > batch_size:
#         raise ValueError('n_jobs must less than batch_size')

#     if isinstance(rowdata, pd.Series):
#         rowdata = rowdata.values

#     # sample batch
#     queue = []
#     for b in range(idx.size - 1):
#         queue.append(rowdata[idx[b]:idx[b + 1]])

#     # multiprocess
#     start_time = datetime.now()
#     print('========================== Start ==========================')
#     pool = Pool(n_jobs)
#     res = pool.starmap(onehot_matrix, zip(queue, repeat(colnums)))
#     pool.close()
#     pool.join()
#     mat = sparse.vstack(res)
#     if save_sparse is False:
#         mat = mat.toarray()
#     print('========================== Time cost: {0} =========================='.format(datetime.now() - start_time))

#     return mat
