from gaussian_quadrature import Hermite
from atomic_orbital import AtomicOrbital
import numpy as np

class HarmonicOscillator(AtomicOrbital):
	def __init__(self, shells, w):
		'''
		shells : maximum principle quantum number of basis
		w :      oscillator frequency
		'''
		N = shells*(shells + 1)
		super().__init__(N, np.identity(N, dtype=float))
		self._H = Hermite(2*shells + 2)	# Hermite Gaussian Quadrature
		self._num_shells = shells
		self._w = w			
		self._sqrt_w = np.sqrt(w)
		self._ww = w*w
		self._create_shells()
		self._compute_normalization_factors()
		self.__create_V_AS()


	def __call__(self, alpha, x, y):
		'''
		evaluate the wave function
		alpha : int symbol for all quantum numbers
		'''
		Hnx = self._H(self._nx[alpha], self._sqrt_w*x)
		Hny = self._H(self._ny[alpha], self._sqrt_w*y)
		return Hnx*Hny*np.exp(-0.5*self._w*(x*x + y*y))*self._norm[alpha]

	# override
	def energy(self, alpha):
		return (self._nx[alpha] + self._ny[alpha] + 1)*self._w

	# override
	def V(self, alpha, beta, gamma, delta):
		if ((self._spin[alpha] + self._spin[gamma]) & 1) or ((self._spin[beta] + self._spin[delta]) & 1):
			return 0
		x1 = y1 = self._H.x
		w1 = self._H.w
		self._H.set_n(self._H.n - 1)
		x2 = y2 = self._H.x
		w2 = self._H.w
		self._H.set_n(self._H.n + 1)

		Hab_x = w1*self._H(self._nx[alpha], x1)*self._H(self._nx[gamma], x1)
		Hab_y = w1*self._H(self._ny[alpha], y1)*self._H(self._ny[gamma], y1)
		Hgd_x = w2*self._H(self._nx[beta], x2)*self._H(self._nx[delta], x2)
		Hgd_y = w2*self._H(self._ny[beta], y2)*self._H(self._ny[delta], y2)

		I = 0.0 
		N1, N2 = len(x1), len(x2)
		for i in range(N1):
			for j in range(N1):
				Hab_ij = Hab_x[i]*Hab_y[j]
				for k in range(N2):
					dx = x1[i] - x2[k]
					for l in range(N2):
						dy = y1[j] - y2[l]
						I += Hab_ij*Hgd_x[k]*Hgd_y[l]/np.sqrt(dx*dx + dy*dy)
		return I/self._ww*self._sqrt_w*self._norm[alpha]*self._norm[beta]*self._norm[gamma]*self._norm[delta]

	# override
	def V_AS(self, alpha, beta, gamma, delta):
		return self._V_AS[alpha, beta, gamma, delta]

	@property
	def w(self):
		'''
		oscillator frequency
		'''
		return self._w


	def __create_V_AS(self):
		'''
		create matrix for storing the antisymmetric coulomb integrals
		'''
		self._V_AS = np.zeros((self._N, self._N, self._N, self._N))
		x1 = y1 = self._H.x
		w1 = self._H.w
		self._H.set_n(self._H.n - 1)
		x2 = y2 = self._H.x
		w2 = self._H.w
		self._H.set_n(self._H.n + 1)

		N1, N2 = len(x1), len(x2)
		v = np.zeros((N1, N1, N2, N2))
		for i in range(N1):
			for j in range(N1):
				for k in range(N2):
					dx = x1[i] - x2[k]
					dy = y1[j] - y2
					v[i, j, k] = 1/np.sqrt((dx*dx + dy*dy))
		v *= self._sqrt_w
		for alpha in range(self._N):
			Ha_x = w1*self._H(self._nx[alpha], x1)
			Ha_y = w1*self._H(self._ny[alpha], y1)
			for beta in range(alpha):
				Hb_x = w2*self._H(self._nx[beta], x2)
				Hb_y = w2*self._H(self._ny[beta], y2)
				for gamma in range(self._N):
					Sag = ((self._spin[alpha] + self._spin[gamma] + 1) & 1)
					Sbg = ((self._spin[beta] + self._spin[gamma] + 1) & 1)
					Hab_x = Ha_x*self._H(self._nx[gamma], x1)
					Hab_y = Ha_y*self._H(self._ny[gamma], y1)

					Hbg_x = Hb_x*self._H(self._nx[gamma], x2)
					Hbg_y = Hb_y*self._H(self._ny[gamma], y2)
					for delta in range(gamma):
						Sabgd = Sag*((self._spin[beta] + self._spin[delta] + 1) & 1)
						Sabdg = Sbg*((self._spin[alpha] + self._spin[delta] + 1) & 1)
						if (Sabgd + Sabdg) == 0:
							continue
						Hgd_x = Hb_x*self._H(self._nx[delta], x2)
						Hgd_y = Hb_y*self._H(self._ny[delta], y2)
						Hps_x = Ha_x*self._H(self._nx[delta], x1)
						Hps_y = Ha_y*self._H(self._ny[delta], y1)
						Vabgd = Vabdg = 0
						for i in range(N1):
							for j in range(N1):
								Hpr = Hab_x[i]*Hab_y[j]
								Hps = Hps_x[i]*Hps_y[j]
								for k in range(N2):
									Vabgd += np.sum(Hpr*Hgd_x[k]*Hgd_y*v[i, j, k, :])
									Vabdg += np.sum(Hps*Hbg_x[k]*Hbg_y*v[i, j, k, :])
						V_AS = (Vabgd*Sabgd - Vabdg*Sabdg)
						normalization = self._norm[alpha]*self._norm[beta]*self._norm[gamma]*self._norm[delta]
						self._V_AS[alpha, beta, gamma, delta] = V_AS*normalization
						self._V_AS[alpha, beta, delta, gamma] = -self._V_AS[alpha, beta, gamma, delta]
						self._V_AS[beta, alpha, gamma, delta] =  self._V_AS[alpha, beta, delta, gamma]
						self._V_AS[beta, alpha, delta, gamma] =  self._V_AS[alpha, beta, gamma, delta]
		self._V_AS/self._ww

	
	def _create_shells(self):
		'''
		create quantum numbers
		'''
		self._nx = np.zeros(self._N, dtype=int)
		self._ny = np.zeros(self._N, dtype=int)
		self._shells = np.zeros(self._num_shells+1, dtype=int)
		self._spin = np.zeros(self._N, dtype=int)
		end = 0
		ns = np.linspace(0, self._num_shells-1, self._num_shells, dtype=int)
		for n in range(self._num_shells):
			start = end
			end += n + 1
			self._spin[start:end] = 1
			self._nx[start:end] = ns[:n+1]
			self._nx[end:end+n+1] = ns[:n+1]
			end += n + 1
			self._ny[start:end] = n - self._nx[start:end]
			self._shells[n+1] = end


	def _compute_normalization_factors(self):
		'''
		compute normalization factor of the AOs
		'''
		self._norm = np.zeros(self._N)
		for alpha in range(self._N):
			nx = self._nx[alpha]
			ny = self._ny[alpha]
			self._norm[alpha] = 1/np.sqrt(self._H.integral(nx)*self._H.integral(ny))


	
	def print_orbitals(self):
		'''
		print out orbital energy and quantum numbers n, nx and ny for each AO
		'''
		print("  n |  E/w | (nx, ny)");
		for i in range(self._num_shells):
			start = self._shells[i]
			end = (start + self._shells[i+1])//2
			print("%3d | %.2lf |  " %(i+1, self.energy(start)/self._w), end="");
			for j in range(start, end):
				print("(%d, %d) " %(self._nx[j], self._ny[j]), end="")
			print()


	def print_spin_orbitals(self):
		'''
		print out orbital energy and quantum numbers n, nx, ny and spin for each AO
		'''
		print("  n |  E/w | (nx, ny)");
		for i in range(self._num_shells):
			start = self._shells[i]
			end = self._shells[i+1]
			print("%3d | %.2lf |  " %(i+1, self.energy(start)/self._w), end="")
			for j in range(start, end):
				if j > self._N:
					break
				sign = "+" if self._spin[j] == 1 else "-"
				print("(%d, %d)%s " %(self._nx[j], self._ny[j], sign), end="")
			print()



if __name__ == "__main__":
	from hartree_fock import HartreeFock
	shells = 3
	w = 1
	psi = HarmonicOscillator(shells, w)
	psi.print_spin_orbitals()
	Ne = [2, 6, 12, 20]
	N = np.min([4, shells])
	for i in range(N):
		HF = HartreeFock(psi, Ne[i])
		counter = HF.solve()
		print(f"N = {Ne[i]:2d} : HF energy is {HF.energy(): 6.4f}. Took {counter} iterations")
	'''
	(base) Eirills-MBP:hartree_fock eirillsh$ python harmonic_oscillator.py 
	  n |  E/w | (nx, ny)
	  1 | 1.00 |  (0, 0)+ (0, 0)- 
	  2 | 2.00 |  (0, 1)+ (1, 0)+ (0, 1)- (1, 0)- 
	  3 | 3.00 |  (0, 2)+ (1, 1)+ (2, 0)+ (0, 2)- (1, 1)- (2, 0)- 
	  4 | 4.00 |  (0, 3)+ (1, 2)+ (2, 1)+ (3, 0)+ (0, 3)- (1, 2)- (2, 1)- (3, 0)- 
	  5 | 5.00 |  (0, 4)+ (1, 3)+ (2, 2)+ (3, 1)+ (4, 0)+ (0, 4)- (1, 3)- (2, 2)- (3, 1)- (4, 0)- 
	N =  2 is 3.023. Took 5 iterations
	N =  6 is 20.18. Took 7 iterations
	N = 12 is 66.05. Took 9 iterations
	N = 20 is 164.7. Took 100 iterations
	'''
