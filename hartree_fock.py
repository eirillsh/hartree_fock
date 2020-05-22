import numpy as np

class HartreeFock:
	'''
	Class for Hartree-Fock calculations
	Limited to orthonormal AO basis
	'''

	def __init__(self, basis, Ne):
		'''
		basis : Atomic Orbital basis
		Ne :    Number of electrons
		'''
		self._Ne = Ne
		self._basis = basis
		self._C = np.identity(self.basis.N, dtype=float)
		self._epsilon = basis.energy(np.linspace(0, self.basis.N-1, self.basis.N))


	def binding_energy(self):
		'''
		the binding energy using MOs
		'''
		D = self.density_matrix()
		E = E0 = 0.0
		for alpha in range(self.basis.N):
			E0 += D[alpha][alpha]*self.basis.energy(alpha)
			for beta in range(self.basis.N):
				Dab = D[alpha][beta]
				for gamma in range(self.basis.N):
					for delta in range(self.basis.N):
						E += Dab*D[gamma][delta]*self.basis.V_AS(alpha, gamma, beta, delta)
		return E0 + 0.5*E


	
	def AO_binding_energy(self):
		'''
		the binding energy using AOs
		'''
		C = self._C 
		self._C = np.identity(self.basis.N, dtype=float)
		E = self.binding_energy()
		self._C = C
		return E


	def solve(self, tol=1e-12, max_it=100):
		'''
		optimize the energy with respect to the coefficients
		'''
		eps_prev = self._epsilon
		self._epsilon, self._C = np.linalg.eigh(self.Fock_matrix())
		counter = 0
		while (np.linalg.norm(eps_prev - self._epsilon) > tol and counter < max_it):
			eps_prev = self._epsilon
			self._epsilon, self._C = np.linalg.eigh(self.Fock_matrix())
			counter += 1
		return counter


	def density_matrix(self):
		'''
		calculate and return the density matrix
		'''
		D = np.zeros((self.basis.N, self.basis.N))
		for alpha in range(self.basis.N):
			for beta in range(self.basis.N):
				D[alpha][beta] = np.sum(self.C[alpha][:self.Ne]*self.C[beta][:self.Ne])
				#D[alpha][beta] = np.sum(self.C[:self.Ne, alpha]*self.C[:self.Ne, beta])
		return D


	def Fock_matrix(self):
		'''
		calculate and return the Fock matrix
		'''
		D = self.density_matrix()
		F = np.zeros((self.basis.N, self.basis.N))
		for alpha in range(self.basis.N):
			F[alpha][alpha] = self.basis.energy(alpha)
			for beta in range(self.basis.N):
				for gamma in range(self.basis.N):
					for delta in range(self.basis.N):
						F[alpha][beta] += D[gamma][delta]*self.basis.V_AS(alpha, gamma, beta, delta)
		return F


	def V(self, p, q, r, s):
		V = 0.0
		for alpha in range(self.basis.N):
			cp = self._C[alpha][p]
			#cp = self._C[p, alpha]
			for beta in range(self.basis.N):
				cp_cr = cp*self._C[beta][r]
				#cp_cr = cp*self._C[r, beta]
				for gamma in range(self.basis.N):
					cp_cq_cr = cp_cr*self._C[gamma][q]
					#cp_cq_cr = cp_cr*self._C[q, gamma]
					for delta in range(self.basis.N):
						cp_cq_cr_cs = cp_cq_cr*self._C[delta][s]
						#cp_cq_cr_cs = cp_cq_cr*self._C[s, delta]
						V += cp_cq_cr_cs*self.basis.V(alpha, gamma, beta, delta)
		return V



	def V_AS(self, p, q, r, s):
		'''
		antisymmetric coulomb integral
		<p q|v|r s> - <p q|v|s r>
		all variables : int symbol for MO
		'''
		V_AS = 0.0
		for alpha in range(self.basis.N):
			cp = self._C[alpha][p]
			#cp = self._C[p, alpha]
			for beta in range(self.basis.N):
				cp_cr = cp*self._C[beta][r]
				#cp_cr = cp*self._C[r, beta]
				for gamma in range(self.basis.N):
					cp_cq_cr = cp_cr*self._C[gamma][q]
					#cp_cq_cr = cp_cr*self._C[q, gamma]
					for delta in range(self.basis.N):
						cp_cq_cr_cs = cp_cq_cr*self._C[delta][s]
						#cp_cq_cr_cs = cp_cq_cr*self._C[s, delta]
						V_AS += cp_cq_cr_cs*self.basis.V_AS(alpha, gamma, beta, delta)
		return V_AS

	
	@property
	def MO_energies(self):
		'''
		Molecular Orbital energies
		'''
		return self._epsilon
	

	@property
	def basis(self):
		'''
		Atomic Orbital Basis
		'''
		return self._basis

	@property
	def Ne(self):
		'''
		Number of electrons
		'''
		return self._Ne

	@property
	def C(self):
		'''
		Coefficient matrix
		'''
		return self._C
	


