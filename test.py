from gaussian_quadrature import Hermite
from hartree_fock import HartreeFock
from moeller_plesset import MoellerPlesset as MøllerPlesset
from coupled_cluster import CoupledCluster
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

def test_V_AS_AO():
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
test_V_AS_AO()




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

def test_HF(HF):
	C = HF.C
	F = HF.Fock_matrix()
	N = len(HF.C)
	error = 0.0
	for i in range(N):
		c = C[:, i]
		diff = np.linalg.norm(F@c - c*HF.MO_energies[i])
		assert diff < 1e-12
		error += diff
	error /= N
	print(f"\tMean of |Fc - c epsilon| = {error:.4g} when using {HF.basis} AOs with {HF.Ne} electrons.")
	assert error < 1e-12, "|Fc - c epsilon| should be zero. Test of HF failed"



print("\nTesting Hartree-Fock")

basis_He = HydrogenLike(2)
basis_Be = HydrogenLike(4)
HF_He = HartreeFock(basis_He, 2)
HF_Be = HartreeFock(basis_Be, 4)
HF_He.solve()
HF_Be.solve()
test_coeff_matrix(HF_He)
test_coeff_matrix(HF_Be)
test_Fock_matrix(HF_He)
test_Fock_matrix(HF_Be)
test_HF(HF_He)
test_HF(HF_Be)





#------TESTING MØLLER-PLESSET------

print("\nTesting Møller-Plesset")

def test_symmetry_V_AS_MO(HF):
	MSE = 0.0
	tol = 1e-12
	for p in range(HF.basis.N):
		for q in range(p):
			for r in range(HF.basis.N):
				diff_pqrr = np.abs(HF.V_AS(p, q, r, r))
				assert diff_pqrr < tol, f"error: V_AS({p, q, r, r}) =  {diff_pqrr :.2g}. Must be zero!"
				diff_pprr = np.abs(HF.V_AS(p, p, r, r))
				assert diff_pprr < tol, f"error: V_AS({p, p, r, r}) =  {diff_pprr :.2g}. Must be zero!"
				for s in range(r):
					diff_pprs = np.abs(HF.V_AS(p, p, r, s))
					assert diff_pprs < tol, f"error: V_AS({p, p, r, s}) =  {diff_pprs :.2g}. Must be zero!"
					V_AS = HF.V_AS(p, q, r, s) 
					diff_pqsr = np.abs(V_AS + HF.V_AS(p, q, s, r))
					assert diff_pqsr < tol, f"error: V_AS({p, q, r, s}) != -V_AS({p, q, s, r}). Difference of {diff_pqsr :.2g}."
					diff_qprs = np.abs(V_AS + HF.V_AS(q, p, r, s))
					assert diff_qprs < tol, f"error: V_AS({p, q, r, s}) != -V_AS({q, p, r, s}). Difference of {diff_qprs :.2g}."
					diff_qpsr = np.abs(V_AS - HF.V_AS(q, p, s, r))
					assert diff_qpsr < tol, f"error: V_AS({p, q, r, s}) !=  V_AS({q, p, s, r}). Difference of {diff_qpsr :.2g}."

					diff = diff_pqrr + diff_pprr + diff_pprs + diff_pqsr + diff_qprs + diff_qpsr
					MSE += diff*diff
	MSE *= 3/HF.basis.N**4
	print(f"\tMSE of error in symmetry of MOs V_AS was {MSE:.4g}. Using Hydrogen AO with {HF.Ne} electrons")
	assert MSE < 1e-12, "MO V_AS symmetry test failed"


def test_V_AS_MO(HF):
	MSE = 0.0
	tol = 10
	for p in range(HF.basis.N):
		for q in range(p):
			for r in range(HF.basis.N):
				for s in range(r):
					V_AS = HF.V(p, q, r, s) - HF.V(p, q, s, r)
					diff = np.abs(HF.V_AS(p, q, r, s) -  V_AS)
					assert diff < tol, f"error: V_AS({p, q, r, s}) !=  V({p, q, r, s}) - V({p, q, s, r}). Difference of {diff:.2g}."
					MSE += diff*diff
	MSE *= 2/HF.basis.N**4
	print(f"\tMSE of error in MOs V_AS was {MSE:.4g}. Using Hydrogen AO with {HF.Ne} electrons")
	#assert MSE < 1e-12, "MO V_AS symmetry test failed"


test_symmetry_V_AS_MO(HF_He)
test_symmetry_V_AS_MO(HF_Be)

test_V_AS_MO(HF_He)
test_V_AS_MO(HF_Be)



#------TESTING COUPLED-CLUSTER------

print("\nTesting Coupled-Cluster")

def test_inital_amplitudes(basis, Z):
	MP = MøllerPlesset(basis, Z)
	MP.solve()
	CCD = CoupledCluster(basis, Z)
	CCD.solve(eta=1.0, max_it=1)

	diff = abs(MP.binding_energy() - CCD.binding_energy())
	print(f"\tDifference between MP energy and CCD with one iteration is {diff:.4g} using {Z} electrons.")
	assert diff < 1e-16, "Difference between MP energy and CCD with one iteration should be zero."


test_inital_amplitudes(basis_He, 2)
test_inital_amplitudes(basis_Be, 4)








