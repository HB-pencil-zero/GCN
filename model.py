import torch 
import dgl 
from layers import *
from torch import Tensor

class GCN:
    def __init__(self,n_feature:int,n_hidden:int,n_class:int,layer_num:int=3)-> None:
        self.layer=[]
        self.n_feature,self.n_hidden,self.n_class,self.layer_num=\
            n_feature,n_hidden,n_class,layer_num
        self.layer.append(graphConvLayer(n_feature,n_hidden))
        for i in range(layer_num-2):
            self.layer.append(graphConvLayer(n_hidden,n_hidden))
        self.layer.append(graphConvLayer(n_hidden,n_class))
    
    def forward(self,x:Tensor)->Tensor:
        for i in range(self.layer_num):
            x=self.layer[i](x)
        return x 
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + ' ('+ \
              str(self.n_feature)+','+str(self.n_hidden)+','+str(self.n_class)+') '
        