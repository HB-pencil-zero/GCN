from turtle import forward
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor

class graphConvLayer(nn.Module):
    def __init__(self,in_features:int,out_features:int,bias:bool=True)->None:
        super(graphConvLayer,self).__init__()
        self.in_features,self.out_features= in_features,out_features

        self.weight=Parameter(data=torch.FloatTensor(self.in_features,self.out_features))
        self.bias=Parameter(torch.FloatTensor(self.out_features))
        if(not bias):
            self.register_parameter("bias",None)
        self.initialise()

    
    def initialise(self):
        stdv=self.weight.size(1)
        self.weight.data.uniform_(-stdv,stdv)
        if(self.bias is not None):
            self.bias.data.uniform_(-stdv,stdv)
    
    def forward(self,x:Tensor,adj:Tensor)->Tensor:
        result =torch.mm(x,self.weight)
        result=torch.spmm(adj,result)
        if(self.bias is not None):
            result=result+self.bias
        return result 
        
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'