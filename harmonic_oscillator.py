import numpy as np
from gaussian_quadrature import Hermite

class HarmonicOscillator:
	def __init__(self, N, w):
		self.N = int(N)		# number of electrons
		self._w = w			# oscillator frequency
		self.num_shells = int(np.sqrt(1 + 4*self.N) - 1)//2
		self.sqrt_w = np.sqrt(w)
		self._compute_shells()
		self.H = Hermite(2*(self.num_shells))


	def __call__(self, i, x, y):
		Hnx = self.H(self.nx[i], self.sqrt_w*x)
		Hny = self.H(self.ny[i], self.sqrt_w*y)
		return Hnx*Hny*np.exp(-self.w*(x*x + y*y)/2.0)

	def energy(self, i):
		return (self.nx[i] + self.ny[i] + 1)*self.w

	@property
	def w(self):
		return self._w
	
	def V(self, p, q, r, s):
		# change to mesh later
		x = y = self.H.x
		w = self.H.w
		N = len(x)
		pr_x = w*self.H(self.nx[p], x)*self.H(self.nx[r], x)
		pr_y = w*self.H(self.ny[p], y)*self.H(self.ny[r], y)
		qs_x = w*self.H(self.nx[q], x)*self.H(self.nx[s], x)
		qs_y = w*self.H(self.ny[q], y)*self.H(self.ny[s], y)
		I = 0.0 
		I_fix = 0.0
		for i in range(N):
			for j in range(N):
				pr_i = pr_x[i]*pr_y[j]
				for k in range(N):
					dx = x[i] - x[k]
					for l in range(N):
						dy = y[j] - y[l]
						r = np.sqrt(dx*dx + dy*dy)
						if r != 0:
							I += pr_i*qs_x[k]*qs_y[l]/r
						I_fix += pr_i*qs_x[k]*qs_y[l]/(r if r != 0 else 1e-16)

		print(I, I_fix)			
		return I

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
				sign = "+" if self.spin[j] == 1 else "-"
				print("(%d, %d)%s " %(self.nx[j], self.ny[j], sign), end="")
			print()



if __name__ == "__main__":
	psi = HarmonicOscillator(2, 0.5)
	psi.print_shells()
	print()
	psi.print_shells_full()

	print("Integrate:")
	print(psi.V(0, 0, 0, 0))


	print((0+1) & 1)
	print((1+1) & 1)
	print((2+1) & 1)
