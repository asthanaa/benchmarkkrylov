# Chemistry Glossary

This glossary keeps the vocabulary in the rest of the docs short and consistent.

## Active space

The subset of orbitals and electrons that you choose to treat explicitly in the correlated calculation.

Example:
- `CAS(6e,6o)` means 6 active electrons in 6 active orbitals

## CAS

Short for **Complete Active Space**.

Inside the chosen active space, all allowed electron arrangements are included. This makes CAS a natural reference problem for comparing approximate solvers.

## Determinant

A Slater determinant is one possible electron-occupation pattern built from spin orbitals.

In this codebase, a CI vector is a weighted sum of determinants.

## Krylov vector

A trial vector used to build a small subspace in which the Hamiltonian is approximately diagonalized.

In plain language:
- start from one vector
- apply the Hamiltonian
- orthogonalize
- repeat

The sequence of vectors forms the Krylov space.

## Ritz value / Ritz vector

The approximate eigenvalue and eigenvector obtained after diagonalizing the Hamiltonian in the current Krylov subspace.

In practice:
- the Ritz value is the current energy estimate
- the Ritz vector tells you the best combination of current basis vectors

## Thick restart

A restart strategy that keeps several good Ritz vectors instead of throwing away the whole subspace.

This usually makes the solver more stable and efficient than restarting from only one vector.

## Reference energy

The comparison target used to judge an approximate method.

In the small teaching examples here, the reference is usually **exact CAS-FCI**.
In the larger advanced workflows, the reference may be **DMRG**.

## Sparse vs dense

- **Dense** means you keep all amplitudes in the working vector
- **Sparse** means you keep only the important amplitudes

CCIK is useful because it tries to keep the important parts of the wavefunction while avoiding the cost of a full dense calculation.
