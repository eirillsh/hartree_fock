from atomic_orbital import AtomicOrbital
import numpy as np

class HydrogenLike(AtomicOrbital):
	def __init__(self, Z):
		'''
		Z : nuclei charge (atmic number)
		'''	
		super().__init__(6, np.identity(6, dtype=float))
		self._Z = Z
		self._ZZ = Z*Z
		self._V = np.load("coulomb/coulomb.npy")*Z
		self.__create_V_AS()


	def calc_V_AS(self, alpha, beta, gamma, delta):
		'''
		calculate antisymmetric coulomb integral
		'''
		Sabgd = ((alpha + gamma + 1) & 1)*((beta + delta + 1) & 1)
		Sabdg = ((alpha + delta + 1) & 1)*((beta + gamma + 1) & 1)
		alpha, beta, gamma, delta = alpha//2, beta//2, gamma//2, delta//2
		Vabgd = self._V[alpha, beta, gamma, delta]*Sabgd
		Vabdg = self._V[alpha, beta, delta, gamma]*Sabdg
		return Vabgd - Vabdg


	# override
	def V_AS(self, alpha, beta, gamma, delta):
		return self._V_AS[alpha, beta, gamma, delta]

	# override
	def V(self, alpha, beta, gamma, delta):
		return self._V[alpha//2, beta//2, gamma//2, delta//2]

	# override
	def energy(self, alpha):
		n = alpha//2 + 1
		return -0.5*self._ZZ/(n*n)

	@property
	def Z(self):
		'''
		atomic number
		'''
		return self._Z

	# override
	def __str__(self):
		return "Hydrogen-like"

	# override
	def __repr__(self):
		return "Hydrogen-like"

	def __create_V_AS(self):
		'''
		create matrix for storing the antisymmetric coulomb integrals
		'''
		self._V_AS = np.zeros((self._N, self._N, self._N, self._N))
		for alpha in range(0, self._N, 2):
			for beta in range(0, self._N, 2):
				for gamma in range(0, self._N, 2):
					Vabgg = self._V[alpha//2, beta//2, gamma//2, gamma//2]
					self._V_AS[alpha+1, beta, gamma+1, gamma] =  Vabgg
					self._V_AS[alpha, beta+1, gamma, gamma+1] =  Vabgg
					self._V_AS[alpha+1, beta, gamma, gamma+1] = -Vabgg
					self._V_AS[alpha, beta+1, gamma+1, gamma] = -Vabgg
					for delta in range(0, gamma, 2):
						Vabgd = self._V[alpha//2, beta//2, gamma//2, delta//2]
						Vabdg = self._V[alpha//2, beta//2, delta//2, gamma//2]
					
						self._V_AS[alpha, beta, gamma, delta] = Vabgd - Vabdg
						self._V_AS[alpha+1, beta+1, gamma+1, delta+1] = Vabgd - Vabdg
						self._V_AS[alpha+1, beta, gamma+1, delta] = Vabgd
						self._V_AS[alpha, beta+1, gamma, delta+1] = Vabgd
						self._V_AS[alpha+1, beta, gamma, delta+1] = -Vabdg
						self._V_AS[alpha, beta+1, gamma+1, delta] = -Vabdg
						
						self._V_AS[alpha, beta, delta, gamma] =  Vabdg - Vabgd
						self._V_AS[alpha+1, beta+1, delta+1, gamma+1] = Vabdg - Vabgd
						self._V_AS[alpha+1, beta, delta+1, gamma] = Vabdg
						self._V_AS[alpha, beta+1, delta, gamma+1] = Vabdg
						self._V_AS[alpha+1, beta, delta, gamma+1] = -Vabgd
						self._V_AS[alpha, beta+1, delta+1, gamma] = -Vabgd

