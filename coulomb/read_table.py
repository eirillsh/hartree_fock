import numpy as np

def pqrs(braket):
	braket = braket.split("|")
	bra = braket[0][-2:]
	ket = braket[2][0:2]
	p = int(bra[0]) - 1
	q = int(bra[1]) - 1
	r = int(ket[0]) - 1
	s = int(ket[1]) - 1
	return (p, q, r, s)

def coulomb(integral):
	integral = integral.split()[1].replace("$", "").replace("Z", "")
	integral = integral.replace("\\sqrt{2}",   "*" + str(np.sqrt(2)))
	integral = integral.replace("\\sqrt{3}",   "*" + str(np.sqrt(3)))
	integral = integral.replace("\\sqrt{6}",   "*" + str(np.sqrt(6)))
	integral = integral.replace("\\sqrt{2/3}", "*" + str(np.sqrt(2/3)))
	return eval(integral)

with open("tex_table.txt", "r") as infile:
	lines = infile.readlines()


V = np.zeros((3, 3, 3, 3))
for line in lines[:-1]:
	line = line.split("&")
	V[pqrs(line[0])] = coulomb(line[1])
	V[pqrs(line[2])] = coulomb(line[3])
line = lines[-1].split("&")
V[pqrs(line[0])] = coulomb(line[1])

np.save("coulomb", V)
		 