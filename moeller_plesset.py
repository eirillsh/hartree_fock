from hartree_fock import HartreeFock

class MoellerPlesset(HartreeFock):
	'''
	Class for Møller-Plesset calculations
	Limited to second order correction
	'''
	
	# override
	def binding_energy(self):
		'''
		Møller Plesset energy
		'''
		dE = 0.0
		for i in range(self.Ne):
			for j in range(i):
				e_ij = self._epsilon[i] + self._epsilon[j]
				for a in range(self.Ne, self.basis.N):
					for b in range(self.Ne, a):
						e_ab = self._epsilon[a] + self._epsilon[b]
						dE += self.V_AS(i, j, a, b)*self.V_AS(a, b, i, j)/(e_ij - e_ab)			
		return HartreeFock.binding_energy(self) + dE
