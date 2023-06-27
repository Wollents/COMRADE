import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
import torch_geometric.data


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot


# 
def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset)) 
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = np.squeeze(np.array(data['Class'],dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels,num_classes)
    ano_labels = np.squeeze(np.array(label))


    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels


def load_mat_amazon(dataset, train_rate=0.3, val_rate=0.1):
    """Load .mat dataset."""
    data = sio.loadmat("./dataset/{}.mat".format(dataset)) 
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    labels = [0]
    ano_labels = np.squeeze(np.array(label))


    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_scipy_sparse_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def adj_to_dgl_graph_tensor(adj):
    """Convert adjacency matrix to dgl format."""
    nx_graph = nx.from_numpy_array(np.array(adj.detach().cpu()))
    dgl_graph = dgl.DGLGraph(nx_graph)
    torch_geometric.utils.from_networkx()
    return dgl_graph
def adj_to_pyg_graph(x, adj):
    # adj_array = np.array(adj.detach().cpu())
    # edge_index = sp.csr_matrix(adj_array)
    # edge_index, _ = torch_geometric.utils.from_scipy_sparse_matrix(edge_index)
    edge_index = adj.to_sparse().indices()
    # for i in range(edge_index.shape[1]):
    #     if edge_index[1,i] > 5195:
    #         print('there is ')
    #         print(edge_index[1,i])
    Pygdata = torch_geometric.data.Data(x, edge_index)
    return Pygdata

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*2)
    subv = []
    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*4)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        # print(subv[i])
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    return subv

def generate_rwr_subgraph_test(dgl_graph, subgraph_size, adj, meanDegree):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1

    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=meanDegree * 2)
    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*4)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        # print(subv[i])
        if (len(subv[i]) <= reduced_size) and (retry_time >10):
            subv[i].append(i)
        else:
            degreeList = {}
            print('the subv[i] is ', subv[i])
            for node in subv[i]:
                degree = torch.sum(adj[node,:]) + torch.sum(adj[:,node])
                degreeList[node] = degree
            chooseList = []
            if len(subv[i]) < subgraph_size*2:
                RankList = sorted(degreeList.items(), key=lambda x:x[1], reverse=True)[:len(subv[i])]
            else:
                RankList = sorted(degreeList.items(),key=lambda x:x[1], reverse=True)[:subgraph_size*2]
            for item in RankList:
                chooseList.append(item[0])

            print('the chooseList is',chooseList)
            tmp = []
            if len(subv[i]) < subgraph_size*2:
                for k in range(reduced_size):
                    tmp.append(chooseList[random.randint(0, len(subv[i])-1)])
            else:
                for k in range(reduced_size):
                    tmp.append(chooseList[random.randint(0, subgraph_size * 2 - 1)])
            subv[i] = tmp

            subv[i].append(i)
    return subv


def generate_rwr_subgraph_v2(dgl_graph, subgraph_size, epoch):
    """Generate subgraph with RWR algorithm."""
    restart_prob = 1
    if epoch > 200:
        restart_prob = 0.8
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=restart_prob, max_nodes_per_seed=subgraph_size*2)
    subv = []
    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*4)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= reduced_size) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        # print(subv[i])
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)
    return subv
