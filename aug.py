from numpy.core.fromnumeric import shape
import torch
import copy
import random
import scipy.sparse as sp
import numpy as np
import scipy.io as sio

# 这个是用于diffusion的图，输入是一个matrix，alpha=0.01， eps=0.001
# this is used for to generate diffusion view, which outputs a matrix.
# NOTE the Line 31-38 that show how to generate a diffusion view.




def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]
    A_loop = sp.eye(N) + A
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
    S_tilde = S.multiply(S >= eps)
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec
    return T_S

from utils import load_mat

datasets = ['pubmed', 'Flickr']
## 示例
## NOTE Example
# for item in datasets:
#     adj, features, labels, idx_train, idx_val, \
#     idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(item)
#     diff = gdc(adj, alpha=0.01, eps=0.0001)
#     np.save('diff_A_' + item, diff)

def gen():
    for i in range(len(datasets)):
        print('loading dataset: ', datasets[i])
        data = sio.loadmat("./dataset/{}.mat".format(datasets[i]))
        adj  = data['Network'] if ('Network' in data) else data['A']
        print('generating dataset', datasets[i])
        diff = gdc(adj, alpha=0.01, eps=0.0001)
        np.save('./diff/diff_A_' + datasets[i], diff)
        print('generating '+ datasets[i] + ' finished')


def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)

    return y.div_(order + 1.0).detach_()

def rand_prop(features, dropnode_rate, A, order, cuda):
    n = features.shape[0]
    drop_rate = dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
    features = masks.cuda(cuda) * features.cuda(cuda)
    features = propagate(features, A, order)
    return features