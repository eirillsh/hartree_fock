# Hartree-Fock, Møller-Plesset and Coupled-Cluster Doubles

Project description by UiO FYS4411. 


General Hartree-Fock, second order Møller-Plesset and Coupled-Cluster doubles calculation on Helium and Beryllium using an atomic basis consisting of hydrogen orbitals.  

The class `HartreeFock` is implemented as a solver class for the Hartree-Fock calculations, using the numpy library to diagonalize the Fock matrix. The atomic orbital basis is given as a parameter upon initialisation, making the solver versatile to any instance of a subclass of the class `AtomicOrbital`. The constraint is that solver assumes an orthonormal basis.

For performance reasons, the implementation allows the subclass of `AtomicOrbital`to either store all anti-symmetrical Coulomb integrals or evaluate them on the fly. In the case of `HydrogenLike`, the Hydrogen-like orbitals used in this project, the small size of the basis ensures that the computations are not memory bound. The anti-symmetric Coulomb integrals of the AO are therefore stored.   

The post-Hartree methods are implemented as subclasses of `HartreeFock`, leaving most of the implementation in the super class. The anti-symmetric Coulomb integrals of the MO are stored to improve performance, as there should be no shortage of memory.



A somewhat rouge harmonic oscillator basis is also included, where the coulomb integrals are numerically integrated using gaussian quadrature. 