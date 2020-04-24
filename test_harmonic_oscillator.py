from harmonic_oscillator import HarmonicOscillator

psi = HarmonicOscillator(30, 0.5)
psi.print_shells()
print()
psi.print_shells_full()

print("Integrate:")
print(psi.integrate(1, 2, 3, 4))