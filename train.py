from model import *
from layers import *
from torch.nn import CrossEntropyLoss
import torch.optim as optim

def train_model(model:GCN,epochs:int,traing_loader,validation_loader,optimizer:optim.SGD,criterion = CrossEntropyLoss)->None:
    for i in range(epochs):
        accu=0
        for data,labels in traing_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(data,labels)
            loss.backward()
            optimizer.step()
            accu=torch.where(torch.argmax(pred,dim=1)== labels).count()/labels.size(0)
            print('epoch {} , loss is {} ,train accuracy is %{} '.format(i,loss,accu*100) )


        for data,labels in traing_loader:
            with torch.no_grad():
                pred = model(data)
                loss = criterion(data,labels)
                accu=torch.where(torch.argmax(pred,dim=1)== labels).count()/labels.size(0)
                print('epoch {} , loss is {},train accuracy is %{} '.format(i,loss,accu*100) )     