from hartree_fock import HartreeFock

class MoellerPlesset(HartreeFock):
	'''
	Class for Møller-Plesset calculations
	Limited to second order correction
	'''
	
	# override
	def bonding_energy(self):
		dE = 0.0
		for i in range(self.Ne):
			for j in range(i):
				e_ij = self._epsilon[i] + self._epsilon[j]
				for a in range(self.Ne, self.basis.N):
					for b in range(self.Ne, a):
						V_AS = self._V_AS(i, j, a, b)
						e_ab = self._epsilon[a] + self._epsilon[b]
						dE += V_AS*V_AS/(e_ij - e_ab)			
		return HartreeFock.bonding_energy(self) + dE

	
	def _V_AS(self, i, j, a, b):
		'''
		antisymmetric coulomb integral
		<i j|v|a b> - <i j|v|b a>
		all variables : int symbol for MO
		'''
		V_AS = 0.0
		for alpha in range(self.basis.N):
			ci = self.C[alpha][i]
			for beta in range(self.basis.N):
				cica = ci*self.C[beta][a]
				for gamma in range(self.basis.N):
					cicjca = cica*self.C[gamma][j]
					for delta in range(self.basis.N):
						cicjcacb = cicjca*cica*self.C[delta][b]
						V_AS += cicjcacb*self.basis.V_AS(alpha, gamma, beta, delta)
		return V_AS
