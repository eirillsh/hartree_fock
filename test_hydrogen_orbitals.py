from hydrogen_like import HydrogenOrbital 


def test_V_AS():
	HO = HydrogenOrbital(2)
	for p in range(HO.N):
		for q in range(HO.N):
			for r in range(HO.N):
				for s in range(HO.N):
					stored = HO.V_AS(p, q, r, s)
					calc = HO.calc_V_AS(p, q, r, s)
					if abs(stored - calc) > 1e-16:
						print(p, q, r, s, stored, calc)


test_V_AS()


'''
print("FOCK MATRIX")
for i in range(6):
	for j in range(6):
		print("%6.2f " %F[i][j], end="")
	print()

print("D")
D = He.density_matrix()
for i in range(6):
	for j in range(6):
		print("%6.2f " %D[i][j], end="")
	print()

print("C")
C = He.C
for i in range(6):
	for j in range(6):
		print("%6.2f " %C[i][j], end="")
	print()

print("I ?")
I = C@C.T
for i in range(6):
	for j in range(6):
		print("%6.2f " %I[i][j], end="")
	print()
'''