from abc import abstractmethod
from obstacles import obstacle
import numpy as np

class Ellipsoid(obstacle.Obstacle):

	origin=np.array([0,0])
	radius=0
	magnitude=0
	rolloff=1

	def __init__(self, origin, radius,magnitude,rolloff,beta):
		super(obstacle.Obstacle, self).__init__()
		self.origin=np.array(origin)
		self.radius=radius
		self.magnitude=magnitude
		self.rolloff=rolloff
		self.beta=beta

	def f(self):
	        return 'hello world'

	def display(self):
		
		print(self.origin)
		print(self.radius)
		print(self.magnitude)
		print(self.rolloff)		
		
	def repel(self,position,velocity,a):
		y=0
		cosine=np.sum(velocity*position,axis=a)/(np.linalg.norm(velocity)*(1+(np.linalg.norm((position-self.origin)/self.radius,axis =a)**2)**self.rolloff))
		print(cosine)
		y=self.magnitude*((-cosine)**self.beta)*(np.linalg.norm(velocity)/(1+(np.linalg.norm((position-self.origin)/self.radius,axis =a)**2)**self.rolloff))/(1+(np.linalg.norm((position-self.origin)/self.radius,axis =a)**2)**self.rolloff)
		return y
