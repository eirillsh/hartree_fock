from harmonic_oscillator import HarmonicOscillator
from hydrogen_like import HydrogenOrbital

psi = HarmonicOscillator(2, 0.5)
psi.print_shells()
print()
psi.print_shells_full()

print("Integrate:")
print(psi.V(0, 0, 0, 0))

print("jh")
psi1 = HydrogenOrbital(2)
print(psi1.V(0, 0, 0, 0))

print((0+1) & 1)
print((1+1) & 1)
print((2+1) & 1)