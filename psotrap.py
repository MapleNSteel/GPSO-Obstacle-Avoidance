import numpy as np
import math
import numdifftools as nd
from gpso import gpso
from obstacles import *

import matplotlib.pyplot as plt

swarmSize=64                       # number of the swarm particles
maxIter=100                        # maximum number of iterations
deltaTime=0.1
inertia=1.61803398875
initialInertia=1.61803398875
socialFactor=1.61803398875*(deltaTime/0.01)
globalFactor=1.61803398875*(deltaTime/0.01)
learningRate=1.61803398875
decayDistance=1
maximise=1

destination=np.array([180,400])
initialPosition = np.array([0,0])
obs=np.array([[50,111.111111111111,30],[300,200,50],[120,340,40]])
obstacleList=[ellipsoid.Ellipsoid(obs[0][0:2],(30,30),20000,10,2),ellipsoid.Ellipsoid(obs[1][0:2],(50,50),20000,10,2),ellipsoid.Ellipsoid(obs[2][0:2],(40,40),20000,10,2)]

x = np.arange(0, 450, 1)
y = np.arange(0, 450, 1)
X, Y = np.meshgrid(x, y)

Z = 0.5*((X-destination[0])**2 + (Y-destination[1])**2)+40000./(1+(((X-obs[0][0])**2+(Y-obs[0][1])**2)/(30**2))**10)+40000./(1+(((X-obs[1][0])**2+(Y-obs[1][1])**2)/(50**2))**10)+15000./(1+(((X-obs[2][0])**2+(Y-obs[2][1])**2)/(60**2))**10);


def objectivefunction(x,v):
	try:
		y = np.linalg.norm(x-destination, axis=1)/2
		for obstacle in obstacleList:
			y+=obstacle.repel(x,v,1)
		return y

	except:
		y = np.linalg.norm(x-destination)/2
		for obstacle in obstacleList:
			y+=obstacle.repel(x,v,0)
		return y



[minima, v]=gpso.optimise(initialPosition=initialPosition,swarmSize=swarmSize,maxIter=maxIter,initialInertia=inertia,socialFactor
=socialFactor,globalFactor=globalFactor,maxSpeedGBest=1800*(0.01/deltaTime),maxSpeedNeighbours=3600*(0.01/deltaTime),learningRate=learningRate
,decayDistance=decayDistance,deltaTime=deltaTime,minimise=True,objectivefunction=objectivefunction)


print (np.asarray(minima))
print (np.asarray(objectivefunction(minima, v)))

plt.figure()
CS = plt.contour(X, Y, Z)
plt.plot(minima[:,0], minima[:,1], 'ro')
plt.figure()
plt.plot(np.asarray(objectivefunction(minima, v)))
plt.axis([-30, 420, -30, 420])
plt.show()
