ccik plugin for Quantum Package
===============================

Repository note
---------------

This plugin source is vendored inside the main ``CompressK`` repository under
``ccik_qp/``. The earlier standalone upstream repository was:

- ``https://github.com/asthanaa/ccik_qp``

The vendored copy is the canonical tracked source for this repository snapshot.

Purpose
-------

This plugin is the production Quantum Package backend for the determinant-sparse
compressed Krylov method implemented in this repository.

The maintained backend is matvec-first:

- candidate discovery is based on connected singles/doubles
- projected amplitudes are evaluated against the full current ``q_k`` support
- external ranking is based on ``|H q_k|`` or Arnoldi-residual amplitudes
- compression happens after orthogonalization

Implemented solver modes
------------------------

- baseline dense CCIK
- thick-restart CCIK

The generalized Ritz solver is unchanged relative to the earlier plugin work.
The Python dense reference path now stabilizes the generalized Ritz solve by
projecting out near-singular overlap modes; the QP plugin continues to use the
same projected ``H_K``/``S_K`` construction and should be validated through the
donor-backed parity workflow before production use after any plugin edit.

Current projected-matvec backends
---------------------------------

``legacy_pt2``
  Legacy compatibility backend. Uses the older CIPSI/PT2-style discovery path.

``connected_sd_fullq``
  Maintained backend. Builds candidates from explicit connected singles/doubles
  off the capped parent set and evaluates projected amplitudes against the full
  current wavefunction support.

Current selector modes
----------------------

- ``abs_hq``
- ``sq_hq``
- ``abs_residual``
- ``sq_residual``
- ``pt2_residual`` (``pt2_legacy`` remains a compatibility alias)

Default maintained selector:

- ``abs_residual``

Compression policies
--------------------

``topk``
  Legacy size-capped mode using ``ccik_selector_keep``, ``ccik_external_keep``,
  ``ccik_nkeep``, and related caps.

``threshold``
  Alternate threshold-only mode. User-facing compression controls are:

  - ``ccik_store_abs_tol``
  - ``ccik_propagate_abs_tol``

Threshold semantics:

- propagation support is defined by ``|q_{k,J}| >= ccik_propagate_abs_tol``
- storage is defined by ``|r_{k,I}| >= ccik_store_abs_tol`` after
  orthogonalization
- if thresholding removes all coefficients, the largest entry is kept as a
  safeguard
- threshold mode forces ``connected_sd_fullq``
- if internal candidate/support safety limits are exceeded, the code aborts with
  an explicit diagnostic instead of silently falling back to top-k truncation

``fixed_budget``
  Fixed-budget sparse mode. User-facing compression controls are:

  - ``ccik_fixed_store_keep``
  - ``ccik_fixed_propagate_keep``

Fixed-budget semantics:

- propagation support is the top ``ccik_fixed_propagate_keep`` coefficients of
  the current normalized stored vector for the legacy selectors
- ``pt2_residual`` is a single-pass fixed-budget selector:
  - build the projected step once and orthogonalize once
  - form PT2 scores
    ``|r_{k,I}|^2 / max(|E_k - H_{II}|, eps)``
    from that one-pass residual
  - choose ``ccik_fixed_store_keep`` from that score map for the next stored
    Krylov vector
  - choose ``ccik_fixed_propagate_keep`` from the same PT2 score family, store
    it on the next Krylov vector, and use it on the following iteration
  - the first PT2-residual iteration falls back to the same propagation rule
    used by ``abs_residual`` because no delayed propagation support exists yet
- next Krylov vectors, thick-restart retained vectors, and the final lowest
  Ritz vector are all compressed to top ``ccik_fixed_store_keep``
- fixed-budget mode forces ``connected_sd_fullq``

Diagnostics
-----------

The plugin can emit detailed per-step traces via:

- ``ccik_debug_trace = True``
- ``ccik_debug_trace_topn = <N>``

The trace includes:

- projected-``Hq`` metadata
- compression policy
- propagation fallback information
- next-vector diagnostics
- top coefficients for projected vectors and next vectors

Rebuild caveat
--------------

The installed plugin tree inside ``QP_ROOT`` is separate from the vendored
workspace copy under this repository. To avoid stale-binary confusion:

- sync the workspace plugin source into the installed plugin tree
- rebuild the installed plugin
- then run parity/validation

The maintained Python tooling already automates this through:

- ``scripts/qp_ccik_common.py``
- ``scripts/debug_selector_parity.py``
- ``scripts/qp_ccik_bridge.py rebuild-plugin``

Canonical validation artifacts for the current plugin path are:

- ``experiments/donor_n2_parity_sweep_20260320_rerun``
- ``experiments/threshold_mode_validation_20260324_release3``

Files
-----

- ``NEED``
  Quantum Package module dependencies
- ``EZFIO.cfg``
  plugin input parameters
- ``ccik.irp.f``
  plugin entry point
- ``ccik_class.irp.f``
  shared sparse-vector storage helpers
- ``ccik_selection_providers.irp.f``
  propagation/generator support providers
- ``ccik_sparse.f90``
  sparse determinant-vector helper routines
- ``ccik_driver.irp.f``
  banner, projected-matvec construction, next-vector generation, and driver
