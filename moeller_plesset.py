from hartree_fock import HartreeFock
import numpy as np

class MoellerPlesset(HartreeFock):
	'''
	Class for Møller-Plesset calculations
	Limited to second order correction

	Methods require that solve() is called first.
	'''
	
	# override
	def binding_energy(self):
		'''
		Møller Plesset energy
		'''	
		E_HF = MP2 = 0.0
		for alpha in range(self.basis.N):
			D_alpha = np.sum(self.C[alpha][:self.Ne]*self.C[alpha][:self.Ne])
			E_HF += D_alpha*self.basis.energy(alpha)

		for i in range(self.Ne):
			for j in range(i):
				E_HF += self._V_AS[i, j, i, j]
				e_ij = self._epsilon[i] + self._epsilon[j]
				for a in range(self.Ne, self.basis.N):
					e_ab = self._epsilon[a] + self._epsilon[self.Ne:a]
					MP2 += np.sum(self._V_AS[a, self.Ne:a, i, j]*self._V_AS[i, j, a, self.Ne:a]/(e_ij - e_ab))
		return E_HF + MP2


	# override
	def solve(self, tol=1e-12, max_it=100):
		HartreeFock.solve(self, tol=tol, max_it=max_it)
		self._compute_V_AS()


	def MP0(self):
		'''
		unperturbed energy
		'''
		return np.sum(self._epsilon[:self.Ne])

	
	def MP1(self):
		'''
		first correction
		'''
		return HartreeFock.binding_energy(self) - self.MP0()

	
	def MP2(self):
		'''
		second correction
		'''
		MP2 = 0.0
		for i in range(self.Ne):
			for j in range(i):
				e_ij = self._epsilon[i] + self._epsilon[j]
				for a in range(self.Ne, self.basis.N):
					e_ab = self._epsilon[a] + self._epsilon[self.Ne:a]
					MP2 += np.sum(self._V_AS[a, self.Ne:a, i, j]*self._V_AS[i, j, a, self.Ne:a]/(e_ij - e_ab))
		return MP2



