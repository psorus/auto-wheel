import numpy
from numpy.random import randint as rndi


def genrnd(n):
    ret= rndi(0,256,size=(n,3)).astype("float")/255
    ret[0,:]=[0.0,0.0,0.0]
    return ret


if __name__=="__main__":
    print(genrnd(5))


