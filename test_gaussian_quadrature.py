from gaussian_quadrature import Hermite
import numpy as np

def test_integrator(b, c):
	n_max = 10
	integrator = Hermite(n_max)
	f = lambda x : np.exp(b*x + c)
	expected = np.sqrt(np.pi)*np.exp(b*b/4.0 + c)
	for n in range(1, n_max+1):
		integrator.set_n(n)
		computed = integrator.integrate(f)
		error = computed - expected
		print("I : %.2f. Error: %.3g" %(computed, error))


test_integrator(0, 0)
print()
test_integrator(1, 0)
print()
test_integrator(1, 0.5)