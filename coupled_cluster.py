from hartree_fock import HartreeFock
import numpy as np

class CoupledCluster(HartreeFock):
	'''
	Class for Coupled Cluster calculations
	Limited to Coupled Cluster Doubles

	Methods require that solve() is called first.
	'''

	def __init__(self, basis, Ne):
		super().__init__(basis, Ne)
		self.virtual = basis.N - Ne
		self._t = np.zeros((Ne, Ne, self.virtual, self.virtual))
		self._d_epsilon = np.zeros((Ne, Ne, self.virtual, self.virtual))


	# override
	def binding_energy(self):
		'''
		Coupled Cluster energy
		'''	
		E_HF = CCD = 0.0
		for alpha in range(self.basis.N):
			D_alpha = np.sum(self.C[alpha][:self.Ne]*self.C[alpha][:self.Ne])
			E_HF += D_alpha*self.basis.energy(alpha)

		for i in range(self.Ne):
			for j in range(i):
				E_HF += self._V_AS[i, j, i, j]
				for a in range(self.Ne, self.basis.N):
					a_idx = a - self.Ne
					CCD += np.sum(self._t[i, j, a_idx, :a_idx]*self._V_AS[i, j, a, self.Ne:a])
		return E_HF + CCD

	
	# override
	def solve(self, eta=1.0, tol=1e-16, max_it=100, HF_tol=1e-12, HF_max_it=100):
		'''
		eta  : learning rate (use in case of oscillation)
		tol    : tolerance of convergence for the amplitudes
		max_it : maximum number of iterations for the amplitude
		HF_tol    : tolerance of convergence for Hartree-Fock
		HF_max_it : maximum number of iterations for Hartree-Fock
		solve Hartree-Fock and optimize the amplitudes
		'''
		HartreeFock.solve(self, tol=HF_tol, max_it=HF_max_it)
		self._compute_V_AS()
		self.compute_d_epsilon()

		t_next = np.inf
		counter = t = 0
		while (counter < max_it and np.linalg.norm(t - t_next) > tol):
			t = self._t
			t_next = self.next_amplitudes()
			self._t = eta*t_next + (1.0 - eta)*self._t
			counter += 1
		return counter 


	def next_amplitudes(self):
		'''
		calculate the next cluster amplitudes 
		t^{n+1}
		'''
		t = np.zeros((self.Ne, self.Ne, self.virtual, self.virtual))
		for i in range(self.Ne):
			for j in range(i):
				for a in range(self.Ne, self.basis.N):
					a_idx = a - self.Ne
					for b in range(self.Ne, a):
						b_idx = b - self.Ne
						t_ij_ab = self.next_amplitude(i, j, a, b)
						t[i, j, a_idx, b_idx] =  t_ij_ab
						t[j, i, b_idx, a_idx] =  t_ij_ab
						t[j, i, a_idx, b_idx] = -t_ij_ab
						t[i, j, b_idx, a_idx] = -t_ij_ab
		return t


	def next_amplitude(self, i, j, a, b):
		'''
		i, j : labels of occupied AOs
		a, b : labels of virtual AOs
		(t_{i j}^{a b})^{n+1}
		calculate the next cluster amplitude
		'''
		a_idx = a - self.Ne
		b_idx = b - self.Ne

		t_ij_ab = self._V_AS[a, b, i, j]

		t_ij_ab += 0.5*np.sum(self._V_AS[a, b, self.Ne:, self.Ne:]*self._t[i, j, :, :])
		t_ij_ab += 0.5*np.sum(self._V_AS[:self.Ne, :self.Ne, i, j]*self._t[:, :, a_idx, b_idx])

		t_ij_ab += np.sum(self._V_AS[:self.Ne, b, self.Ne:, j]*self._t[i, :, a_idx, :])
		t_ij_ab -= np.sum(self._V_AS[:self.Ne, b, self.Ne:, i]*self._t[j, :, a_idx, :])
		t_ij_ab -= np.sum(self._V_AS[:self.Ne, a, self.Ne:, j]*self._t[i, :, b_idx, :])
		t_ij_ab += np.sum(self._V_AS[:self.Ne, a, self.Ne:, i]*self._t[j, :, b_idx, :])

		for k in range(self.Ne):
			for l in range(k):
				for c in range(self.Ne, self.basis.N):
					c_idx = c - self.Ne

					coeff  = self._t[i, j, c_idx, :c_idx]*self._t[k, l, a_idx, b_idx]

					coeff += 4*self._t[i, k, a_idx, c_idx]*self._t[j, l, b_idx, :c_idx]
					coeff -= 4*self._t[j, k, a_idx, c_idx]*self._t[i, l, b_idx, :c_idx]

					coeff -= 2*self._t[i, k, :c_idx, c_idx]*self._t[l, j, a_idx, b_idx]
					coeff += 2*self._t[j, k, :c_idx, c_idx]*self._t[l, i, a_idx, b_idx]

					coeff -= 2*self._t[l, k, a_idx, c_idx]*self._t[i, j, :c_idx, b_idx]
					coeff += 2*self._t[l, k, b_idx, c_idx]*self._t[i, j, :c_idx, a_idx]
					t_ij_ab += np.sum(coeff*self._V_AS[k, l, c, self.Ne:c])
		return t_ij_ab/self._d_epsilon[i, j, a_idx, b_idx]


	def compute_d_epsilon(self):
		'''
		compute all permutations of
		(epsilon_i + epsilon_j - epsilon_a - epsilon_b)
		epsilon : MO energy 
		i, j : label of occupied MO
		a, b : label of virtual MO
		'''
		for i in range(self.Ne):
			self._d_epsilon[i, :, :, :] += self._epsilon[i]
			self._d_epsilon[:, i, :, :] += self._epsilon[i]
		for a in range(self.Ne, self.basis.N):
			a_idx = a - self.Ne
			self._d_epsilon[:, :, a_idx,  : ] -= self._epsilon[a]
			self._d_epsilon[:, :,   :, a_idx] -= self._epsilon[a]


	def t(self, i, j, a, b):
		'''
		i, j : labels of occupied AOs
		a, b : labels of virtual AOs
		t_{i j}^{a b}
		cluster amplitude 
		'''
		return self._t[i, j, a - self.Ne, b - self.Ne]


	