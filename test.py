from gaussian_quadrature import Hermite
from hartree_fock import HartreeFock
from moeller_plesset import MoellerPlesset as MøllerPlesset
from hydrogen_like import HydrogenLike
import numpy as np

verbose = True


#------TESTING GAUSSIAN QUADRATURE------

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
	print(f"\tMSE of Gaussian Quadrature was {MSE:.4g}.")
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
		print("\tTesting Hermite with known integral")
		print(f"\t- relative error ({b1}, {c1}): {error1:.3g}")
		print(f"\t- relative error ({b2}, {c2}): {error2:.3g}")
		print(f"\t- relative error ({b3}, {c3}): {error3:.3g}")

	gaussian_quadrature_polynomial(GQ)


print("Testing Gaussian Quadrature:")
test_hermite()





#------TESTING HYDROGEN ORBITALS------

def test_V_AS():
	HO = HydrogenLike(2)
	MSE = 0.0
	for p in range(HO.N):
		for q in range(HO.N):
			for r in range(HO.N):
				for s in range(HO.N):
					diff = np.abs(HO.V_AS(p, q, r, s) -  HO.calc_V_AS(p, q, r, s))
					assert diff < 1e-16, f"error in matrix-form V_AS({p, q, r, s})"
					MSE += diff*diff
	MSE /= HO.N**4 
	print(f"\tMSE of matrix form V_AS was {MSE:.4g}.")
	assert MSE < 1e-12, "Hydrogen Orbitals matrix form V_AS failed test"


print("\nTesting Hydrogen Obritals")
test_V_AS()




#------TESTING HARTREE-FOCK------


def test_coeff_matrix(HF):
	C = HF.C
	norm = np.linalg.norm(C@C.T - np.identity(HF.basis.N))
	print(f"\t|C@C.T - I| = {norm:.4g} when using {HF.basis} AOs with {HF.Ne} electrons.")
	assert norm < 1e-12, "|C@C.T - I| should be zero. Test of HF failed"


def test_Fock_matrix(HF):
	F = HF.Fock_matrix()
	norm = np.linalg.norm(F - F.T)
	print(f"\t|F - F.T| = {norm:.4g} when using {HF.basis} AOs with {HF.Ne} electrons.")
	assert norm < 1e-12, "|F - F.T| should be zero. Test of HF failed"



print("\nTesting Hartree-Fock")

He = HartreeFock(HydrogenLike(2), 2)
Be = HartreeFock(HydrogenLike(4), 4)
He.solve()
Be.solve()
test_coeff_matrix(He)
test_coeff_matrix(Be)
test_Fock_matrix(He)
test_Fock_matrix(Be)














