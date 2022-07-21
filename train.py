from model import *
from layers import *
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import argparse
import numpy as np 
from utils import *
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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch:int):
    model.train()
    optimizer.zero_grad()
    output=model(adj@features)
    loss=CrossEntropyLoss(output[idx_train],labels[idx_train])
    optimizer.step()
    accu=accuracy(output[idx_train],labels[idx_train])
    print("epoch {} train finished , and accuracy is {}".format(epoch+1,accu))


def test(epoch:int):
    model.eval()
    with torch.no_grad():
        output=model(adj@features)
        loss=CrossEntropyLoss(output[idx_val],labels[idx_val])
        accu=accuracy(output[idx_val],labels[idx_val])
        print("epoch {} test, and accuracy is {}".format(epoch+1,accu))

if __name__ == '__main__':
    epochs=args.epochs
    for i in range(epochs):
        train(epoch=i)
        test(epoch=i)
        print("*" for i in range(10))
    print("finished all epochs train")
    
