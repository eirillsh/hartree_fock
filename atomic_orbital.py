
class AtomicOrbital:
	'''
	representing atomic orbital basis
	used for Hartree-Fock calculations
	'''
	def __init__(self, N, S):
		'''
		N : size of basis
		S : overlap matrix
		'''
		self._N = N
		self._S = S

	def V(self, alpha, beta, gamma, delta):
		'''
		coloumb integral
		<alpha beta|v|gamma delta>
		all variables : int symbol for all quantum numbers
		'''
		raise NotImplementedError



	def V_AS(self ,alpha, beta, gamma, delta):
		'''
		antisymmetric coulomb integral
		<alpha beta|v|gamma delta> - <alpha beta|v|delta gamma>
		all variables : int symbol for all quantum numbers
		'''
		raise NotImplementedError


	def energy(self, alpha):
		'''
		one-electron energy
		alpha : int symbol for all quantum numbers
		'''
		raise NotImplementedError


	@property
	def N(self):
		'''
		size of basis
		'''
		return self._N

	@property
	def S(self):
		'''
		overlap matrix
		'''
		return self._S


	def __str__(self):
		return "Atomic Orbital Basis"


	def __repr__(self):
		return "Atomic Orbital Basis"
