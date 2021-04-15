#import matplotlib.pyplot as plt
from plt import *
import numpy as np

from tqdm import tqdm


outdim=2
import sys
if len(sys.argv)>1:
    outdim=int(sys.argv[1])

f=np.load(f"out{outdim}.npz")

x=f["x"]
p=f["p"]

n=100000


d=np.mean(p**2,axis=-1)
i=np.argmin(d)

print([int(zw*256) for zw in x[i]])
print(p[i])



x=x[:n]
p=p[:n]




for xx,pp in tqdm(zip(x,p),total=len(x)):
    plt.plot([pp[0]],[pp[1]],".",color=xx,alpha=1.0)

plt.savefig(f"colorwheel{outdim}.png",format="png")
plt.savefig(f"colorwheel{outdim}.pdf",format="png")

#plt.plot(np.arange(10),color=x[1])
plt.show()





