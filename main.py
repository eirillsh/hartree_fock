from hartree_fock import HartreeFock
from moeller_plesset import MoellerPlesset as MøllerPlesset
from hydrogen_like import HydrogenLike
import numpy as np 
import matplotlib.pyplot as plt



def plot_MO_energies(E, atom, Z):
	x = range(len(E))
	y_tick = [E[i] for i in range(0, len(E), 2)]
	plt.plot(x[:Z], E[:Z], ".", color="black", markersize=15)
	plt.plot(x[Z:], E[Z:], ".", color="darkgrey", markersize=15)
	plt.xticks(x, [r"$\psi_%d$" %i for i in x], fontsize=16)
	plt.yticks(y_tick, ["%.3f" %y for y in y_tick], fontsize=15)
	plt.grid()
	plt.xlabel(r"$\psi_i$", fontsize=17)
	plt.ylabel(r"MO energy [a.u]", fontsize=17)
	plt.tight_layout()
	plt.savefig("results/" + atom + ".pdf")
	plt.close()

def plot_AO_energies(basis, atom):
	x = np.linspace(0, 5, 6, dtype=int)
	y_tick = [basis.energy(i) for i in range(0, 6, 2)]
	x_tick  = [r"$\phi_{1s\uparrow}$", r"$\phi_{1s\downarrow}$"]
	x_tick += [r"$\phi_{2s\uparrow}$", r"$\phi_{2s\downarrow}$"]
	x_tick += [r"$\phi_{3s\uparrow}$", r"$\phi_{3s\downarrow}$"]
	plt.plot(x, basis.energy(x), ".", color="black", markersize=15)
	plt.xticks(x, x_tick, fontsize=17)
	plt.yticks(y_tick, ["%.3f" %y for y in y_tick], fontsize=15)
	plt.grid()
	plt.xlabel(r"$\phi_\alpha$", fontsize=16)
	if atom == "H":
		plt.ylabel(r"AO energy $\frac{\mathrm{[a.u]}}{Z^2}$", fontsize=17)
	else:
		plt.ylabel(r"AO energy [a.u]", fontsize=17)
	plt.tight_layout()
	plt.savefig("results/AO_" + atom + ".pdf")
	plt.close()


with open("results/table.txt", "w") as outfile:
	outfile.write("Bonding Energy:\n")
	print("\nHELIUM")
	Z = 2
	orbitals = HydrogenLike(Z)
	He = HartreeFock(orbitals, Z)
	He_phi = He.bonding_energy()
	print(f"Before HF      energy is {He_phi}")
	He.solve()
	He_HF = He.bonding_energy()
	print(f"Hartree-Fock   energy is {He_HF}")

	He = MøllerPlesset(orbitals, Z)
	He.solve()
	He_MP = He.bonding_energy()
	print(f"Møller-Plesset energy is {He_MP}")
	print(f"Experimental   energy is -2.904")
	outfile.write("\t\t$\\mathrm{He}$ & %.4f & %.4f & %.4f & -2.904\\\\" % (He_phi, He_HF, He_MP))


	print("\n\nBERYLLIUM")
	Z = 4
	orbitals = HydrogenLike(Z)
	Be = HartreeFock(orbitals, Z)
	Be_phi = Be.bonding_energy()
	print(f"Before HF      energy is {Be_phi}")
	Be.solve()
	Be_HF = Be.bonding_energy()
	print(f"Hartree-Fock   energy is {Be_HF}")

	Be = MøllerPlesset(orbitals, Z)
	Be.solve()
	Be_MP = Be.bonding_energy()
	print(f"Møller-Plesset energy is {Be_MP}")
	print(f"Experimental   energy is -14.67")
	outfile.write("\n\t\t$\\mathrm{Be}$ & %.3f & %.3f & %.3f & -14.67" % (Be_phi, Be_HF, Be_MP))

	#outfile.write("\n\nRelative errors:\n")
	E = -2.904
	HF = (E - He_HF)/E*100
	MP = (E - He_MP)/E*100
	#outfile.write("\t\t$\\mathrm{He} $ & %.2f  & %.2f \\\\" % (HF, MP))
	E = -14.67
	HF = (E - Be_HF)/E*100
	MP = (E - Be_MP)/E*100
	#outfile.write("\n\t\t$\\mathrm{Be} $ & %.2f & %.2f " % (HF, MP))




plot_MO_energies(He.MO_energies, "He", 2)
plot_MO_energies(Be.MO_energies, "Be", 4)

plot_AO_energies(HydrogenLike(2), "He")
plot_AO_energies(HydrogenLike(4), "Be")
plot_AO_energies(HydrogenLike(1), "H")




