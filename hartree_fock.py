import numpy as np

class HartreeFock:

	def __init__(self, basis, Ne):
		self._Ne = Ne
		self._basis = basis
		self._C = np.identity(self.basis.N, dtype=float) 


	def energy(self):
		E = E_H0 = 0.0
		D = self.density_matrix()
		E = E0 = 0.0
		for alpha in range(self.basis.N):
			E0 += D[alpha][alpha]*self.basis.energy(alpha)
			for beta in range(self.basis.N):
				Dab = D[alpha][beta]
				for gamma in range(self.basis.N):
					for delta in range(self.basis.N):
						E += Dab*D[gamma][delta]*self.basis.V_AS(alpha, gamma, beta, delta)
		return 0.5*E + E0


	def solve(self, tol=1e-12, max_it=100):
		E_prev, self._C = np.linalg.eigh(self.Fock_matrix())
		E, self._C = np.linalg.eigh(self.Fock_matrix())
		counter = 0
		while (self.MSE(E_prev, E) > tol and counter < max_it):
			E_prev = E
			E, self._C = np.linalg.eigh(self.Fock_matrix())
			counter += 1
		self.E = E
		return counter


	def MSE(self, E_prev, E):
		diff = E_prev - E
		return np.mean(diff*diff)


	def density_matrix(self):
		D = np.zeros((self.basis.N, self.basis.N))
		for alpha in range(self.basis.N):
			for beta in range(self.basis.N):
				D[alpha][beta] = np.sum(self.C[alpha][:self.Ne]*self.C[beta][:self.Ne])
		return D


	def Fock_matrix(self):
		D = self.density_matrix()
		F = np.zeros((self.basis.N, self.basis.N))
		for alpha in range(self.basis.N):
			F[alpha][alpha] = self.basis.energy(alpha)
			for beta in range(self.basis.N):
				for gamma in range(self.basis.N):
					for delta in range(self.basis.N):
						F[alpha][beta] += D[gamma][delta]*self.basis.V_AS(alpha, gamma, beta, delta)
		return F


	@property
	def basis(self):
		return self._basis

	@property
	def Ne(self):
		return self._Ne

	@property
	def C(self):
		return self._C
	


