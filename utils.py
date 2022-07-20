import numpy as np
import scipy.sparse as sp
import torch
a={4:1,5:2,6:3}
b=[4,5,6]
for i in map(a.get,b):
    print (i)