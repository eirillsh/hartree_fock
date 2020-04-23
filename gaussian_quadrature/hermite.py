from .gaussian_quadrature import Gaussian_Quadrature
import numpy as np

class Hermite(Gaussian_Quadrature):

	def __init__(self, n):
		self._sqrt_pi = np.sqrt(np.pi)
		super().__init__(n, 1, n, max_factorial=n)

	def _create_polynomials(self):
		polynomial = [np.array([1], dtype=int), np.array([2, 0], dtype=int)]
		for n in range(1, self.n):
			H_next = np.zeros(n+2, dtype=int)
			H_next[:-1]  += polynomial[n] 
			H_next[2:] -= n*polynomial[n-1]
			polynomial.append(2*H_next)
		return polynomial

	def _create_weights(self):
		n = self._n-1
		H = np.polyval(self.polynomial[n], self._x)
		return self.integral(n)/(H*H*self._n)

	def integral(self, n):
		return ((1 << n)*self._factorial[n])*self._sqrt_pi

	def integration_limits(self):
		return "(-inf, inf)"