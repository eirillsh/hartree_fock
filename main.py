from hartree_fock import HartreeFock
from moeller_plesset import MoellerPlesset as MøllerPlesset
from hydrogen_like import HydrogenOrbital
import numpy as np 

print("\nHELIUM")
Z = 2
orbitals = HydrogenOrbital(Z)
He = HartreeFock(orbitals, Z)
He.solve()
HF = He.energy()

He = MøllerPlesset(orbitals, Z)
He.solve()
MP = He.energy()
print(f"Experimental   energy is -2.904")
print(f"Møller-Plesset energy is {MP}")
print(f"Hartree-Fock   energy is {HF}")




print("\n\nBERYLLIUM")
Z = 4
orbitals = HydrogenOrbital(Z)
Be = HartreeFock(orbitals, Z)
Be.solve()
HF = Be.energy()

Be = MøllerPlesset(orbitals, Z)
Be.solve()
MP = Be.energy()
print(f"Experimental   energy is -14.67")
print(f"Møller-Plesset energy is {MP}")
print(f"Hartree-Fock   energy is {HF}")





