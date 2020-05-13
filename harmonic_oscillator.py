import numpy as np
from gaussian_quadrature import Hermite

class HarmonicOscillator:
	def __init__(self, shells, w):
		self._w = w			# oscillator frequency
		self.num_shells = shells
		self.N = self.num_shells*(self.num_shells + 1)
		self.sqrt_w = np.sqrt(w)
		self.ww = w*w
		self._compute_shells()
		self.H = Hermite(2*self.num_shells)
		self._compute_normalization_factors()
		self.__create_V_AS()


	def __call__(self, i, x, y):
		Hnx = self.H(self.nx[i], self.sqrt_w*x)
		Hny = self.H(self.ny[i], self.sqrt_w*y)
		return Hnx*Hny*np.exp(-self.w*(x*x + y*y)/2.0)

	def energy(self, i):
		return (self.nx[i] + self.ny[i] + 1)*self.w

	@property
	def w(self):
		return self._w

	def V_AS(self, p, q, r, s):
		return self._V_AS[p, q, r, s]


	def __create_V_AS(self):
		self._V_AS = np.zeros((self.N, self.N, self.N, self.N))
		x1 = y1 = self.H.x
		w1 = self.H.w
		N1 = len(x1)

		self.H.set_n(self.H.n - 1)
		x2 = y2 = self.H.x
		w2 = self.H.w
		N2 = len(x2)
		self.H.set_n(self.H.n + 1)
		#x2 = x1; y2 = y1; N2 = N1; w2 = w1

		v = np.zeros((N1, N1, N2, N2))
		for i in range(N1):
			for j in range(N1):
				for k in range(N2):
					dx = x1[i] - x2[k]
					dy = y1[j] - y2
					r = np.sqrt((dx*dx + dy*dy))
					v[i, j, k][r != 0] = 1/r[r != 0]
		v *= self.sqrt_w

		for p in range(self.N):
			Hp_x = w1*self.H(self.nx[p], x1)
			Hp_y = w1*self.H(self.ny[p], y1)
			for q in range(p):
				Hq_x = w2*self.H(self.nx[q], x2)
				Hq_y = w2*self.H(self.ny[q], y2)
				for r in range(self.N):
					Spr = ((self.spin[p] + self.spin[r] + 1) & 1)
					Sqr = ((self.spin[q] + self.spin[r] + 1) & 1)
					Hpr_x = Hp_x*self.H(self.nx[r], x1)
					Hpr_y = Hp_y*self.H(self.ny[r], y1)

					Hqr_x = Hq_x*self.H(self.nx[r], x2)
					Hqr_y = Hq_y*self.H(self.ny[r], y2)
					for s in range(r):
						Spqrs = Spr*((self.spin[q] + self.spin[s] + 1) & 1)
						Spqsr = Sqr*((self.spin[p] + self.spin[s] + 1) & 1)
						if (Spqrs + Spqsr) == 0:
							continue
						Hqs_x = Hq_x*self.H(self.nx[s], x2)
						Hqs_y = Hq_y*self.H(self.ny[s], y2)
						Hps_x = Hp_x*self.H(self.nx[s], x1)
						Hps_y = Hp_y*self.H(self.ny[s], y1)

						Vpqrs = 0
						Vpqsr = 0
						for i in range(N1):
							for j in range(N1):
								Hpr = Hpr_x[i]*Hpr_y[j]
								Hps = Hps_x[i]*Hps_y[j]
								for k in range(N2):
									Vpqrs += np.sum(Hpr*Hqs_x[k]*Hqs_y*v[i, j, k, :])
									Vpqsr += np.sum(Hps*Hqr_x[k]*Hqr_y*v[i, j, k, :])
						V_AS = (Vpqrs*Spqrs - Vpqsr*Spqsr)/self.ww
						normalization = self.norm[p]*self.norm[q]*self.norm[r]*self.norm[s]
						self._V_AS[p, q, r, s] = V_AS*normalization
						self._V_AS[p, q, s, r] = -self._V_AS[p, q, r, s]
						self._V_AS[q, p, r, s] =  self._V_AS[p, q, s, r]
						self._V_AS[q, p, s, r] =  self._V_AS[p, q, r, s]

	
	
	def V(self, p, q, r, s):
		if ((self.spin[p] + self.spin[r]) & 1) or ((self.spin[q] + self.spin[s]) & 1):
			return 0
		x1 = y1 = self.H.x
		w1 = self.H.w
		N1 = len(x1)

		self.H.set_n(self.H.n - 1)
		x2 = y2 = self.H.x
		w2 = self.H.w
		N2 = len(x2)
		self.H.set_n(self.H.n + 1)

		pr_x = w1*self.H(self.nx[p], x1)*self.H(self.nx[r], x1)
		pr_y = w1*self.H(self.ny[p], y1)*self.H(self.ny[r], y1)
		qs_x = w2*self.H(self.nx[q], x2)*self.H(self.nx[s], x2)
		qs_y = w2*self.H(self.ny[q], y2)*self.H(self.ny[s], y2)

		I = 0.0 
		for i in range(N1):
			for j in range(N1):
				pr_ij = pr_x[i]*pr_y[j]
				for k in range(N2):
					dx = x1[i] - x2[k]
					for l in range(N2):
						dy = y1[j] - y2[l]
						r = np.sqrt(dx*dx + dy*dy)
						I += pr_ij*qs_x[k]*qs_y[l]/r
		return I/self.ww

	# only for full shells
	def _compute_shells(self):
		self.nx = np.zeros(self.N, dtype=int)
		self.ny = np.zeros(self.N, dtype=int)
		self.shells = np.zeros(self.num_shells+1, dtype=int)
		self.spin = np.zeros(self.N, dtype=int)
		end = 0
		ns = np.linspace(0, self.num_shells-1, self.num_shells, dtype=int)
		for n in range(self.num_shells):
			start = end
			end += n + 1
			self.spin[start:end] = 1
			self.nx[start:end] = ns[:n+1]
			self.nx[end:end+n+1] = ns[:n+1]
			end += n + 1
			self.ny[start:end] = n - self.nx[start:end]
			self.shells[n+1] = end


	def _compute_normalization_factors(self):
		self.norm = np.zeros(self.N)
		for i in range(self.N):
			nx = self.nx[i]
			ny = self.ny[i]
			self.norm[i] = 1/np.sqrt(self.H.integral(nx)*self.H.integral(ny))


	
	def print_shells(self):
		print("  n |  E/w | (nx, ny)");
		for i in range(self.num_shells):
			start = self.shells[i]
			end = (start + self.shells[i+1])//2
			print("%3d | %.2lf |  " %(i+1, self.energy(start)/self._w), end="");
			for j in range(start, end):
				print("(%d, %d) " %(self.nx[j], self.ny[j]), end="")
			print()


	def print_shells_full(self):
		print("  n |  E/w | (nx, ny)");
		for i in range(self.num_shells):
			start = self.shells[i]
			end = self.shells[i+1]
			print("%3d | %.2lf |  " %(i+1, self.energy(start)/self._w), end="")
			for j in range(start, end):
				if j > self.N:
					break
				sign = "+" if self.spin[j] == 1 else "-"
				print("(%d, %d)%s " %(self.nx[j], self.ny[j], sign), end="")
			print()



if __name__ == "__main__":
	from hartree_fock import HartreeFock
	shells = 3
	w = 1
	psi = HarmonicOscillator(shells, w)
	psi.print_shells_full()
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
