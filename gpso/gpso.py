import numpy as np
from numpy import pi
import math

from gpso.gradient import gradient

def optimise(initialPosition,swarmSize,maxIter,initialInertia,socialFactor,globalFactor
,maxSpeedGBest,maxSpeedNeighbours,learningRate,deltaTime,decayDistance,minimise,objectivefunction,gradient=gradient.gradient):
        swarmPositions=np.random.random((swarmSize,len(initialPosition)))+initialPosition
        swarmVelocities=np.random.random((swarmSize,len(initialPosition)))
        swarmValues=objectivefunction(swarmPositions, swarmVelocities)

        localBestPositions=swarmPositions;
        localBestValues=swarmValues

        gBest=0
        gBestPosition=localBestPositions[gBest,:]
        gBestPositions=np.zeros((maxIter,len(initialPosition)))
        gBestVelocities=np.zeros((maxIter,len(initialPosition)))

        for iter in range(0,maxIter):
                inertia = initialInertia*(1-math.exp(-swarmValues[gBest]/(decayDistance)))
                swarmValues=objectivefunction(swarmPositions, swarmVelocities)
                for j in range(0,swarmSize):
                        if (swarmValues[j]<localBestValues[j] and minimise==True) or (swarmValues[j]>localBestValues[j] and minimise==False):
                                localBestValues[j]=swarmValues[j]
                                localBestPositions[j,:]=swarmPositions[j,:]
                
                        if (swarmValues[j]<swarmValues[gBest] and minimise==True) or (swarmValues[j]>swarmValues[gBest] and minimise==False):
                                gBest=j
        
                gBestPosition=swarmPositions[gBest,:]
                gBestPositions[iter,:]=(gBestPosition)
                gBestVelocities[iter,:]=(swarmVelocities[gBest,:])

                swarmVelocities = inertia*np.random.random((swarmSize,len (initialPosition)))*swarmVelocities + (socialFactor/deltaTime)*np.random.random((swarmSize,len(initialPosition)))*(localBestPositions- swarmPositions) +(globalFactor/deltaTime)*np.random.random((swarmSize,len(initialPosition)))*(gBestPosition - swarmPositions) - learningRate*gradient(objectivefunction,swarmPositions, swarmVelocities) 

		           
                for j in range(0,swarmSize):   
                        if (np.linalg.norm(swarmVelocities[j,:])!=0):
                                if j==gBest:
                                        if np.linalg.norm(swarmVelocities[j,:])**2 >= maxSpeedGBest*maxSpeedGBest and maxSpeedGBest!=0:        
                                                swarmVelocities[j,:]/=np.linalg.norm(swarmVelocities[j,:])
                                                swarmVelocities[j,:]*=maxSpeedGBest
                                else:
                                        if np.linalg.norm(swarmVelocities[j,:])**2 >= maxSpeedNeighbours*maxSpeedNeighbours and np.all(maxSpeedNeighbours)!=0:        
                                                swarmVelocities[j,:]/=np.linalg.norm(swarmVelocities[j,:])
                                                swarmVelocities[j,:]*=maxSpeedNeighbours
		
                swarmPositions=swarmPositions+swarmVelocities*deltaTime

        return [gBestPositions, gBestVelocities]
