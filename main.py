import numpy as np
from rnd import genrnd

numdot=100000
x=np.load("data.npz")["q"][:numdot]#not perfect, but limited memory

#np.random.seed(12)
#np.random.shuffle(x)
#x=x[:numdot]

x=x.astype("float")/255

print(x.shape)

"""
q->x,y=f(q)
std(x)=1,mean(x)=0
std(y)=1,mean(y)=0
abs(f(q[0])-f(q[1]))=0
"""



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras import backend as K

outdim=2
import sys
if len(sys.argv)>1:
    outdim=int(sys.argv[1])

model=keras.Sequential(
        [
            layers.Input((3,)),
            layers.Dense(5),
            layers.Dense(9),
            layers.Dense(5),
            layers.Dense(outdim),
        ]
        )

def loss(out,q):
    #print(q.shape)
    #exit()

    #std=1
    l1=K.mean((K.mean(K.abs(q),axis=(0,1))-1)**2)
    
    #mean=0
    l2=K.mean((K.mean(q,axis=(0,1)))**2)

    #similarity
    delta=K.mean((q[:,0,:]-q[:,1,:])**2)

    #anticorrelation terms

    #code for other dimensions
    ct=0.00001
    cf=None
    for i in range(outdim):
        for j in range(i+1,outdim):
            ac=K.abs(K.mean(q[:,:,i]*q[:,:,j]))
            if cf is None:
                cf=ac
            else:
                cf+=ac
            ct+=1.0
    ucc=cf/ct

    #ucc=K.abs(K.mean(q[:,:,0]*q[:,:,1]))


    return l1+l2+delta**2+ucc


model.compile(optimizer="adam",loss=loss)

model.fit(x,x,epochs=10)


y=genrnd(1000000)
p=model.predict(y,verbose=2)

np.savez_compressed(f"out{outdim}",x=y,p=p)



