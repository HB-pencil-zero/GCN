from cv2 import normalize
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

def encode_onehot(label):
    """
    return on_hot encode for labels 
    """
    classes=set(label)
    class_dict={j: np.identity(len(classes))[i,:] for i,j in enumerate(classes)}
    onehot_code=np.array(list(map(class_dict.get,label)),
                            dtype=np.int32)
    return onehot_code 

def load_data(path="/home/huangbei/GCN/dataset/cora/",dataset="cora"):
    
    print( "loading {} dataset ".format(dataset) )

    idx_features_labels= np.genfromtxt( "{}{}.content".format(path,dataset),
            dtype=np.dtype(str))
    
    features=sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)

    labels=encode_onehot(idx_features_labels[:,-1])

    idx=np.array(idx_features_labels[:,0],dtype=np.int32)
    idx_map={j:i for i,j in enumerate(idx)}

    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print(labels.shape[0])
    adj=sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    
    adj=adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj) #generate a symmetric adj 

    features=normalize(features)
    adj=normalize(adj+sp.eye(idx.shape[0]))
    
    idx_train=torch.LongTensor(range(140))
    idx_val=torch.LongTensor(range(200,500))
    idx_test=torch.LongTensor(range(500,1500))

    return adj,features, labels, idx_train, idx_val, idx_test


"""def load_data(path="/home/huangbei/GCN/dataset/cora/", dataset="cora"):

    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, """

def normalize(mx:np.ndarray):
    row_sum=np.array(mx.sum(1))
    row_inv=np.power(row_sum,-1)
    row_inv[np.where(row_sum==0)]=0
    row_inv=row_inv.squeeze(axis=1)
    row_mat_inv=np.diag(row_inv)
    return row_mat_inv@mx


def accuracy(outputs:torch.Tensor,labels):
    pred=outputs.max(1)[1].type_as(labels)
    correct=pred.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def  sparse_mx_to_torch_sparse_tensor(mx:sp.coo_matrix):
    '''把sparse的matrix转换为sparse tensor
    '''
    mx=mx.tocoo()#.astype(np.float32)
    indice=torch.from_numpy(
        np.vstack(mx.row,mx.col).astype(np.int32)
    )
    values=torch.from_numpy(mx.data)
    shape=torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indice, values, shape).to_dense()