from gaussian_quadrature import Hermite
import numpy as np

verbose = True


def gaussian_quadrature_polynomial(GQ, tol=1e-12):
	MSE = 0.0
	for n in range(GQ.n_max):
		for m in range(GQ.n_max):
			degree = (n + m)//2 + 1
			if degree < GQ.n_min:
			 	degree = GQ.n_min
			elif degree > GQ.n_max:
			 	degree = GQ.n_max
			GQ.set_n(degree)
			f = lambda x: GQ(n, x)*GQ(m, x)
			error = GQ.integrate(f) - (0 if n != m else GQ.integral(n))
			error *= error
			if error > tol and verbose:
				print(f"I({n}, {m}) : {error:.3g}")
			MSE += error
	MSE /= GQ.n_max*GQ.n_max
	print(f"MSE of Gaussian Quadrature was {MSE:.4g}.")
	assert MSE < tol, "Gaussian Quadrature failed test"



def test_hermite():
	b1, b2, b3 = 0, 3, 2
	c1, c2, c3 = 0, 0, 3

	f1 = lambda x : np.exp(b1*x + c1)
	f2 = lambda x : np.exp(b2*x + c2)
	f3 = lambda x : np.exp(b3*x + c3)
	y = lambda b, c: np.sqrt(np.pi)*np.exp(b*b/4.0 + c)

	GQ = Hermite(10)

	error1 =  (GQ.integrate(f1) - y(b1, c1))/y(b1, c1)
	error2 =  (GQ.integrate(f2) - y(b2, c2))/y(b2, c2)
	error3 =  (GQ.integrate(f3) - y(b3, c3))/y(b3, c3)

	if verbose:
		print("\nTesting Hermite with known integral")
		print(f"relative error ({b1}, {c1}): {error1:.3g}")
		print(f"relative error ({b2}, {c2}): {error2:.3g}")
		print(f"relative error ({b3}, {c3}): {error3:.3g}")

	gaussian_quadrature_polynomial(GQ)


test_hermite()



