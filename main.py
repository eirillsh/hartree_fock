from hartree_fock import HartreeFock
from moeller_plesset import MoellerPlesset as MøllerPlesset
from coupled_cluster import CoupledCluster
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


def create_results(atom):
	basis = HydrogenLike(atom['Z'])
	plot_AO_energies(basis, atom['name'])
	HF = HartreeFock(basis, atom['Z'])
	iterations_HF = HF.solve()
	atom["HF"] = HF.binding_energy()
	atom["AO"] = HF.AO_binding_energy()
	plot_MO_energies(HF.MO_energies, atom['name'], atom['Z'])
	MP = MøllerPlesset(basis, atom['Z'])
	MP.solve()
	atom["MP"] = MP.binding_energy()
	CC = CoupledCluster(basis, atom['Z'])
	iterations_CC = CC.solve()
	atom["CC"] = CC.binding_energy()

	HF_dev = 100*(atom["exp"] - atom["HF"])/atom["exp"]
	MP_dev = 100*(atom["exp"] - atom["MP"])/atom["exp"]
	CC_dev = 100*(atom["exp"] - atom["CC"])/atom["exp"]
	print(f"Number of iterations: {iterations_HF} HF and {iterations_CC} CCD.")
	print(f"Before HF       energy is %.4f. Error : %.3f %%." %(atom["AO"], 100*(atom["exp"] - atom["AO"])/atom["exp"]))
	print(f"Hartree-Fock    energy is %.4f. Error : %.3f %%." %(atom["HF"], HF_dev))
	print(f"Møller-Plesset  energy is %.4f. Error : %.3f %%. Improved %.3g %% from HF." %(atom["MP"], MP_dev, HF_dev - MP_dev))
	print(f"Coupled-Cluster energy is %.4f. Error : %.3f %%. Improved %.3g %% from HF." %(atom["CC"], CC_dev, HF_dev - CC_dev))
	print(f"Experimental    energy is %.4f." %(atom["exp"]))
	atom["table"] = "\n\t\t$\\mathrm{%s}$ & %.5g & %.5g & %.5g & %.5g & %.4g\\\\" \
		%(atom['name'], atom["AO"], atom["HF"], atom["MP"],  atom["CC"], atom["exp"])


def smaller_basis_He(N):
	atom = {"name" : "He_basis%d" %N, "Z" : 2, "exp" : -2.904}
	basis = HydrogenLike(atom['Z'])
	basis._N = N
	HF = HartreeFock(basis, atom['Z'])
	iterations_HF = HF.solve()
	atom["HF"] = HF.binding_energy()
	atom["AO"] = HF.AO_binding_energy()
	plot_MO_energies(HF.MO_energies, atom['name'], atom['Z'])
	MP = MøllerPlesset(basis, atom['Z'])
	MP.solve()
	atom["MP"] = MP.binding_energy()
	CC = CoupledCluster(basis, atom['Z'])
	iterations_CC = CC.solve()
	atom["CC"] = CC.binding_energy()

	HF_dev = 100*(atom["exp"] - atom["HF"])/atom["exp"]
	MP_dev = 100*(atom["exp"] - atom["MP"])/atom["exp"]
	CC_dev = 100*(atom["exp"] - atom["CC"])/atom["exp"]
	print(f"Number of iterations: {iterations_HF} HF and {iterations_CC} CCD.")
	print(f"Before HF       energy is %.4f. Error : %.3f %%." %(atom["AO"], 100*(atom["exp"] - atom["AO"])/atom["exp"]))
	print(f"Hartree-Fock    energy is %.4f. Error : %.3f %%." %(atom["HF"], HF_dev))
	print(f"Møller-Plesset  energy is %.4f. Error : %.3f %%. Improved %.3g %% from HF." %(atom["MP"], MP_dev, HF_dev - MP_dev))
	print(f"Coupled-Cluster energy is %.4f. Error : %.3f %%. Improved %.3g %% from HF." %(atom["CC"], CC_dev, HF_dev - CC_dev))
	print(f"Experimental    energy is %.4f." %(atom["exp"]))
	atom["table"] = "\n\t\t$\\%s$ & %.5g & %.5g & %.5g & %.5g & %.4g\\\\" \
		%("reduced", atom["AO"], atom["HF"], atom["MP"],  atom["CC"], atom["exp"])
	return atom
	

with open("results/table.txt", "w") as outfile:
	outfile.write("Binding Energy:")

	print("\nHELIUM")
	He = {"Z": 2, "name" : "He", "exp" : -2.904}
	create_results(He)
	outfile.write(He["table"])

	print("\n\nBERYLLIUM")
	Be = {"Z": 4, "name" : "Be", "exp" : -14.67}
	create_results(Be)
	outfile.write(Be["table"])

	print("\n\nHELIUM SMALLER BASIS")
	He4 = smaller_basis_He(4)
	outfile.write("\nBinding Energy of He with reduced basis:")
	outfile.write(He4["table"])



key = ["AO", "HF", "MP", "CC"]
label = [r"$E^{\Phi}$", r"$E^{HF}$", r"$E^{MP}$", r"$E^{CC}$"]
x = [4, 6]
E = []
for i in range(1, 4):
	y = [(He4[key[i]] - He["exp"])/He["exp"], (He[key[i]] - He["exp"])/He["exp"]]
	E += y
	plt.plot(x, y, "--", marker="o", label=label[i])
plt.legend()
plt.xticks([4, 6], fontsize=16)
plt.yticks(E, [""]*6, fontsize=15)
#plt.axis([3.9, 6.1, He["CC"] - 0.001, He4["HF"] + 0.001])
plt.grid()
plt.savefig("results/compare.pdf")
plt.close()





