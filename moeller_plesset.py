from hartree_fock import HartreeFock

class MoellerPlesset(HartreeFock):

	def energy(self):
		dE = 0.0
		for i in range(self.Ne):
			for j in range(i):
				for a in range(self.Ne, self.basis.N):
					for b in range(self.Ne, a):
						V_AS = self.V_AS(i, j, a, b)
						dE += V_AS*V_AS/(self.E[i] + self.E[j] - self.E[a] - self.E[b])			
		return HartreeFock.energy(self) + dE

	
	def V_AS(self, i, j, a, b):
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
