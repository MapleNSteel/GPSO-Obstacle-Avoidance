import numpy as np
import math
def gradient(func,x, v):
    delx=1e-10
    diff=np.eye(np.shape(x)[1])*delx

    g=np.zeros(np.shape(x))
    for i in range(np.shape(x)[1]):
        g[0:,i]=np.array(((func(x+diff[i,0:], v)-func(x-diff[i,0:], v))/(2*delx)))

    return g

