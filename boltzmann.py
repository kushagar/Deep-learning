import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import variable

#movies=pd.read_csv("ml-1m/movies.dat",sep='::',header=None,engine='python',encoding='latin-1')
#users=pd.read_csv("ml-1m/users.dat",sep='::',header=None,engine='python',encoding='latin-1')
#ratings=pd.read_csv("ml-1m/ratings.dat",sep='::',header=None,engine='python',encoding='latin-1')
training_set=pd.read_csv("ml-100k/u1.base",delimiter='\t')
training_set=np.array(training_set,dtype='int')
test_set=pd.read_csv("ml-100k/u1.test",delimiter="\t")
test_set=np.array(test_set,dtype='int')

nb_users=int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]),max(test_set[:,1])))

def convert(data):
    new_data=[]
    for i in range(nb_users+1):
        id_movies=data[:,1][data[:,0]==i]
        id_ratings=data[:,2][data[:,0]==i]
        ratings=np.zeros(nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append(list(ratings))
    return new_data
training_set=convert(training_set)
test_set=convert(test_set)        
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)
training_set[training_set==0]=-1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1
test_set[test_set==0]=-1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1

class RBM:
    def __init__(self,vn,hn):
        self.W=torch.randn(hn,vn)
        self.a=torch.randn(1,hn)
        self.b=torch.randn(1,vn)
    def sample_h(self,x):
        wx=torch.mm(x,self.W.t())
        activation=wx+self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy=torch.mm(y,self.W)
        activation=wy+self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)   
    def train(self,v0,vk,ph0,phk):
        self.W += (torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)).t()
        self.b+=torch.sum((v0-vk),0)
        self.a+=torch.sum((ph0-phk),0)
nv=len(training_set[0])
nh=100
batchsize=100
model=RBM(nv,nh)

#training the model
epochs=10
for i in range(1,epochs+1):
    train_loss=0
    s=0.0
    for j in range(0,nb_users-batchsize,batchsize):
        vk=training_set[j:j+batchsize]
        v0=training_set[j:j+batchsize]
        ph0,_=model.sample_h(v0)
        for k in range(10):
          _,hk=model.sample_h(vk)
          _,vk=model.sample_v(hk)
          vk[v0<0]=v0[v0<0]
        phk,_=model.sample_h(vk)      
        model.train(v0,vk,ph0,phk)
        train_loss+=torch.mean(torch.abs(v0[v0>0]-vk[v0>0]))
        s+=1
    print('epoch:'+str(i)+' loss:'+str(train_loss/s))    
import statistics

test_loss=0
s=0
for j in range(0,nb_users):
    v=training_set[j:j+1]
    vt=test_set[j:j+1]
    if(len(vt[vt>0])>0):
        _,h=model.sample_h(v)
        _,v=model.sample_v(h)
        test_loss+=torch.mean(torch.abs(vt[vt>0]-v[vt>0]))
        s+=1
print("test_loss:"+str(test_loss/s))    
# _,h=model.sample_h(test_set[0:1])     
# v,_=model.sample_v(h)
# torch.set_printoptions(threshold=1682)
# print(v)
    
