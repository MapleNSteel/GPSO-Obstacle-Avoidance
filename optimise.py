import numpy as np
import math
import numdifftools as ndnumdifftools
from gpso import gpso
from obstacles import *
import matplotlib.pyplot as plt

swarmSize=64                       # number of the swarm particles
maxIter=100                        # maximum number of iterations
deltaTime=0.01
inertia=1.1
initialInertia=1.1
socialFactor=1.1*(deltaTime/0.01)
globalFactor=1.1*(deltaTime/0.01)
terminateDistance=1
learningRate=1.61803398875e-5
maximise=1
decayDistance=1

initialPosition = np.array([0,0])

def objectivefunction(x):
	try:
		y=np.sum(np.sin(x)**2,1)
	except:
		y=np.sum(np.sin(x)**2)
	return y

minima=gpso.optimise(initialPosition,swarmSize,maxIter,inertia,socialFactor,globalFactor,1800*(0.01/deltaTime),3600*(0.01/deltaTime),learningRate,decayDistance,deltaTime,1,objectivefunction)

print ("a"+str(np.asarray(minima)))
print (np.asarray(objectivefunction(minima)))
print (initialInertia*(1-np.exp(-objectivefunction(minima))))

#plt.plot((objectivefunction(minima)))
#plt.plot(initialInertia*(1-np.exp(-objectivefunction(minima))))
#plt.show()
