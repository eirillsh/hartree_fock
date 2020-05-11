import numpy as np

class HydrogenOrbital:
	def __init__(self, Z):	
		self._Z = Z
		self._ZZ = Z*Z
		self._V = np.load("coulomb/coulomb.npy")*Z
		self._N = 6
		self._S = np.identity(self._N, dtype=float) 
		self.__create_V_AS()


	def calc_V_AS(self, p, q, r, s):
		SqrSqs = ((p + r + 1) & 1)*((q + s + 1) & 1)
		SqsSqr = ((p + s + 1) & 1)*((q + r + 1) & 1)
		p, q, r, s = p//2, q//2, r//2, s//2
		Vpqrs = self._V[p, q, r, s]*SqrSqs
		Vpqsr = self._V[p, q, s, r]*SqsSqr
		return Vpqrs - Vpqsr

	def V_AS(self, p, q, r, s):
		return self._V_AS[p, q, r, s]


	def __create_V_AS(self):
		self._V_AS = np.zeros((self._N, self._N, self._N, self._N))
		for p in range(0, self._N, 2):
			for q in range(0, self._N, 2):
				for r in range(0, self._N, 2):
					Vpqrr = self._V[p//2, q//2, r//2, r//2]
					self._V_AS[p+1, q, r+1, r] =  Vpqrr
					self._V_AS[p, q+1, r, r+1] =  Vpqrr
					self._V_AS[p+1, q, r, r+1] = -Vpqrr
					self._V_AS[p, q+1, r+1, r] = -Vpqrr

					for s in range(0, r, 2):
						Vpqrs = self._V[p//2, q//2, r//2, s//2]
						Vpqsr = self._V[p//2, q//2, s//2, r//2]
					
						self._V_AS[p, q, r, s] = Vpqrs - Vpqsr
						self._V_AS[p+1, q+1, r+1, s+1] = Vpqrs - Vpqsr
						self._V_AS[p+1, q, r+1, s] = Vpqrs
						self._V_AS[p, q+1, r, s+1] = Vpqrs
						self._V_AS[p+1, q, r, s+1] = -Vpqsr
						self._V_AS[p, q+1, r+1, s] = -Vpqsr

						self._V_AS[p, q, s, r] =  Vpqsr - Vpqrs
						self._V_AS[p+1, q+1, s+1, r+1] = Vpqsr - Vpqrs
						self._V_AS[p+1, q, s+1, r] = Vpqsr
						self._V_AS[p, q+1, s, r+1] = Vpqsr
						self._V_AS[p+1, q, s, r+1] = -Vpqrs
						self._V_AS[p, q+1, s+1, r] = -Vpqrs




	def V(self, p, q, r, s):
		return self._V[p//2, q//2, r//2, s//2]


	def energy(self, alpha):
		n = alpha//2 + 1
		return -0.5*self._ZZ/(n*n)

	@property
	def Z(self):
		return self._Z

	@property
	def N(self):
		return self._N

	@property
	def S(self):
		return self._S
	
	
	


