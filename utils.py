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
                            dtype=np.float32)
    return onehot_code 

def load_data(path="/home/huangbei/GCN/dataset/cora/",dataset="cora"):
    "function to load sample data  \
    "
    print( "loading {} dataset ".format(dataset) )

    idx_features_labels= np.genfromtxt( "{}{}.content".format(path,dataset),
            dtype=np.dtype(str))
    
    features=sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
    labels=encode_onehot(idx_features_labels[:,-1])
    idx=idx_features_labels[:,0]

    idx_dict={j:i for i,j in enumerate(idx)}

    edges_unorder=np.genfromtxt("{}{}.cites".format(path,dataset),
                                dtype=np.np.int32)
    edges = np.array(list(map(idx_dict,edges_unorder.flatten())),
                        dtype=np.np.int32).reshape(edges_unorder.shape)

    adj=sp.coo_matrix((np.ones(edges.shape[0]),edges[:,0],edges[:,1]),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj=adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj) #generate a symmetric adj 

    features=normalize(features)
    adj=normalize(adj+sp.eye(idx.shape[0]))
    
    idx_train=range(140)
    idx_val=range(200,500)
    idx_test=range(500,1500)

    return adj,features, labels, idx_train, idx_val, idx_test


def normalize(mx:np.ndarray):
    row_sum=np.array(mx.sum(1))
    row_inv=np.power(mx,-1)
    row_inv[np.where(row_sum==0)]=0
    row_mat_inv=np.diag(row_inv)
    return row_mat_inv@mx


def accuracy(outputs:torch.Tensor,labels):
    pred=torch.max(outputs,dim=1)[1].type_as(labels)
    correct=outputs.eq(labels).double()
    correct=correct.sum()
    return correct/len(labels)

def sparse_mx_to_sparse_tensor(mx:sp.coo_matrix):
    '''把sparse的matrix转换为sparse tensor
    '''
    mx=mx.tocoo()#.astype(np.float32)
    indice=torch.from_numpy(
        np.vstack(mx.row,mx.col).astype(np.int32)
    )
    values=torch.from_numpy(mx.data)
    shape=torch.Size(mx.shape)
    return torch.sparse_coo_tensor(indice, values, shape).to_dense()