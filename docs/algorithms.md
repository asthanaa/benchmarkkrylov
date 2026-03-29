# Algorithms

This page is part of the **advanced** track.

If you are new to the repository, read the examples and the glossary before this page.

## High-level idea

All three solver families build a Krylov subspace for the CAS Hamiltonian, but they do it with sparse determinant vectors instead of keeping every determinant amplitude.

The basic loop is:

1. start from a sparse vector
2. approximate `H q_k`
3. orthogonalize against earlier basis vectors
4. keep a compressed next vector
5. solve a small Ritz problem in the built subspace

## Implemented solver families

- `ccik`
  - baseline deterministic solver
- `ccik_thick`
  - deterministic solver with thick restart
- `ccik_stochastic`
  - stochastic candidate-discovery variant

## Compression policies

- `topk`
  - legacy fixed-support truncation
- `threshold`
  - threshold-based truncation
- `fixed_budget`
  - recommended modern policy
  - top-`Kstore` storage
  - top-`Kprop` propagation

## Why `ccik_thick` is the recommended default

Thick restart keeps several useful Ritz vectors between cycles. In practice this gives a better balance of stability and cost than restarting from only one vector.

## Where to read the implementation

- baseline solver: `src/ccik/core.py`
- thick restart: `src/ccik/thick_restart.py`
- stochastic discovery: `src/ccik/stochastic.py`
- selection machinery: `src/ccik/selection.py`
- sparse vector machinery: `src/ccik/sparse_ci.py`
