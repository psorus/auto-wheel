import numpy as np

q=np.load("dataraw.npz")["q"]


q1=np.roll(q,1,axis=1)
q2=np.roll(q,1,axis=2)


def rss(q):
    q=np.reshape(q,[int(np.prod(q.shape[:-1])),q.shape[-1]])
    q=np.reshape(q,[q.shape[0]]+[1]+[q.shape[1]])
    return q

q=rss(q)
q1=rss(q1)
q2=rss(q2)

d1=np.concatenate((q,q1),axis=1)
d2=np.concatenate((q,q2),axis=1)

d=np.concatenate((d1,d2),axis=0)

print(d.shape)

np.savez_compressed("data",q=d)



