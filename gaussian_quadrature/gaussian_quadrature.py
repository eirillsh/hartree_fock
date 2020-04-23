import numpy as np

class Gaussian_Quadrature:
	def __init__(self, n, n_min, n_max, max_factorial=0):
		self._n = n
		self._n_min = n_min
		self._n_max = n_max
		self._create_factorial(max_factorial)
		self.polynomial = self._create_polynomials()
		self._x = np.roots(self.polynomial[self._n])
		self._w = self._create_weights()


	def __call__(self, n, x):
		return np.polyval(self.polynomial[n], x)

	def set_n(self, n):
		if self._n_min <= n <= self._n_max:
			self._n = n
			self._x = np.roots(self.polynomial[self._n])
			self._w = self._create_weights()
		else:
			raise ValueError("n is not within bounds")

	def integrate(self, f):
		return np.sum(self._w*f(self._x))

	def _create_polynomials(self):
		raise NotImplementedError("Gaussian Quadrature needs polynomials")

	def _create_weights(self):
		raise NotImplementedError("Gaussian Quadrature needs weights")

	def _create_factorial(self, max_factorial):
		if max_factorial == 0:
			pass
		self._factorial = np.ones(self.n_max+1, dtype=int)
		for n in range(1, self.n_max):
			self._factorial[n+1] = self._factorial[n]*(n+1)
		self.factorial = lambda n : self._factorial[n]

	def integral(self, n):
		raise NotImplementedError("Gaussian Quadrature needs the integral of its polynomials squared")

	def	normalization_constant(self, n):
		return 1.0/np.sqrt(self.integral(n))

	@property
	def x(self):
		return self._x

	@property
	def w(self):
		return self._w	

	@property
	def n(self):
		return self._n

	@property
	def n_max(self):
		return self._n_max

	@property
	def n_min(self):
		return self._n_min

	def within_bounds(self, n):
		raise NotImplementedError("Gaussian Quadrature needs limits for n")

	def integration_limits(self):
		raise NotImplementedError("Gaussian Quadrature needs integration limits")
	