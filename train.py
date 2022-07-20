from model import *
from layers import *
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import argparse

parser=argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')


args=parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


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