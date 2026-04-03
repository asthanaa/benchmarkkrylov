"""Dataclasses that define solver defaults and accepted parameter names.

These classes are the easiest place to inspect the knobs exposed by the public
API. For most learners, :class:`CCIKThickRestartParams` is the best class to
read first because ``ccik_thick`` is the recommended default solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CCIKParams:
    """Parameters controlling the baseline sparse CCIK iteration.

    Plain-language purpose:
    - collect the baseline solver settings in one immutable object

    Chemistry meaning:
    - controls how the Krylov space is built and how aggressively determinant
      amplitudes are compressed during the calculation
    """

    # Maximum number of Krylov basis vectors to build.
    m: int = 100
    # Legacy top-k knob: number of score-ranked external determinants admitted
    # into the working support update when compression_policy="topk".
    nadd: int = 2000
    # Legacy top-k knob: maximum support size retained for each compressed
    # Krylov vector q_k when compression_policy="topk".
    nkeep: int = 10000
    # Legacy top-k knob: propagation/selector cap used before the final top-k
    # storage compression when compression_policy="topk".
    Kv: int = 3000
    # Compression policy: fixed_budget (default), threshold, or topk (legacy).
    compression_policy: str = "fixed_budget"
    # Absolute threshold for storing normalized Krylov vectors in threshold mode.
    store_abs_tol: float = 0.0
    # Absolute threshold for using normalized q_k coefficients in H q_k propagation.
    propagate_abs_tol: float = 0.0
    # Fixed-budget store cap used when compression_policy="fixed_budget".
    fixed_store_keep: int = 65_536
    # Fixed-budget propagation cap used when compression_policy="fixed_budget".
    fixed_propagate_keep: int = 512
    # Candidate discovery mode: connected_sd, selected_ci (default), stochastic, or full.
    candidate_mode: str = "selected_ci"
    # Projected Ritz policy for deterministic solvers: exact_on_basis or approximate.
    projected_ritz_mode: str = "exact_on_basis"
    # Selector mode used to rank the projected H q_k, its orthogonal residual,
    # or the current lowest Ritz state in fixed-budget mode.
    selector_mode: str = "abs_residual"
    # Number of repeated selected-CI expansion layers used to build each projected H|q_k>.
    selector_depth: int = 1
    # Maximum number of parent determinants used to spawn candidate excitations.
    generator_keep: int = 256
    # Drop parent determinants smaller than this cutoff before candidate expansion.
    generator_abs_coeff_cutoff: float = 0.0
    # Optional hard cap applied after generator_keep.
    max_parents_for_expansion: int | None = None
    # Keep the current q_k support in the projected candidate pool.
    include_current_support: bool = True
    # Oversampling factor used before the final post-orthogonalization compression.
    tmp_keep_factor: int = 4
    # Preserve a capped subset of the current q_k support in the final compression when possible.
    preserve_prev_support: bool = True
    # If None, use min(2048, nkeep // 4) at runtime.
    preserve_prev_support_cap: int | None = None
    # Compute an exact H q_k for tiny validation problems and log the projection error.
    exact_hq_diagnostics: bool = False
    # Selected-CI parent coefficient threshold.
    selector_ci_coeff_cutoff: float = 5e-4
    # Selected-CI determinant generation threshold.
    selector_cutoff: float = 5e-4
    # Emit per-iteration diagnostics.
    verbose: bool = False
    # Treat smaller orthogonalized residual norms as Krylov breakdown.
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CCIKThickRestartParams:
    """Parameters for sparse CCIK with thick restart.

    Plain-language purpose:
    - collect the settings for the recommended thick-restart solver

    Chemistry meaning:
    - controls both the restart schedule and the determinant-compression policy
      used inside each cycle
    """

    # Krylov basis size built within one restart cycle.
    m_cycle: int = 25
    # Maximum number of restart cycles.
    ncycles: int = 10
    # Number of lowest Ritz vectors reused as restart seeds.
    nroot: int = 3
    # Stop when successive restart energies differ by less than tol.
    tol: float = 1e-6

    # Legacy top-k knobs from baseline CCIK. These are only active when
    # compression_policy="topk".
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000
    compression_policy: str = "fixed_budget"
    store_abs_tol: float = 0.0
    propagate_abs_tol: float = 0.0
    fixed_store_keep: int = 65_536
    fixed_propagate_keep: int = 512
    candidate_mode: str = "selected_ci"
    projected_ritz_mode: str = "exact_on_basis"
    # Selector mode used to rank the projected H q_k, its orthogonal residual,
    # or the current lowest Ritz state in fixed-budget mode.
    selector_mode: str = "abs_residual"
    selector_depth: int = 1
    generator_keep: int = 256
    generator_abs_coeff_cutoff: float = 0.0
    max_parents_for_expansion: int | None = None
    include_current_support: bool = True
    tmp_keep_factor: int = 4
    preserve_prev_support: bool = True
    preserve_prev_support_cap: int | None = None
    exact_hq_diagnostics: bool = False
    selector_ci_coeff_cutoff: float = 5e-4
    selector_cutoff: float = 5e-4
    verbose: bool = False
    orth_tol: float = 1e-12


@dataclass(frozen=True)
class CCIKStochasticParams:
    """Parameters for CCIK-stochastic candidate discovery.

    Plain-language purpose:
    - collect the settings for the stochastic discovery variant

    Chemistry meaning:
    - mixes Krylov iteration settings with stochastic proposal settings such as
      walker count and random seed
    """

    # Maximum number of Krylov basis vectors.
    m: int = 100

    # Legacy top-k knobs from baseline CCIK. These are only active when
    # compression_policy="topk".
    nadd: int = 2000
    nkeep: int = 10000
    Kv: int = 3000
    compression_policy: str = "fixed_budget"
    store_abs_tol: float = 0.0
    propagate_abs_tol: float = 0.0
    fixed_store_keep: int = 65_536
    fixed_propagate_keep: int = 512
    # Candidate discovery mode: stochastic (default), connected_sd, selected_ci, or full.
    candidate_mode: str = "stochastic"
    # Selector mode used to rank the projected H q_k or its orthogonal residual.
    # The new ground-state Ritz PT2 selector is deterministic-only and is not
    # supported for the stochastic solver.
    selector_mode: str = "abs_residual"
    # Number of fixed-budget stochastic discovery / projected-contraction rounds
    # when candidate_mode='stochastic', or repeated selected-CI expansion layers
    # when candidate_mode='selected_ci'.
    selector_depth: int = 1
    generator_keep: int = 256
    generator_abs_coeff_cutoff: float = 0.0
    max_parents_for_expansion: int | None = None
    include_current_support: bool = True
    tmp_keep_factor: int = 4
    preserve_prev_support: bool = True
    preserve_prev_support_cap: int | None = None
    exact_hq_diagnostics: bool = False
    # Selected-CI parent coefficient threshold.
    selector_ci_coeff_cutoff: float = 5e-4
    # Selected-CI determinant generation threshold.
    selector_cutoff: float = 5e-4

    # Number of stochastic proposals used to discover candidates for each iteration.
    n_walkers: int = 20000
    # Random seed for reproducible candidate discovery. Use None for nondeterministic runs.
    seed: int | None = 0

    # Parents are sampled with probability proportional to |q_J|**parent_power.
    parent_power: float = 1.0
    # Probability of proposing a double excitation instead of a single excitation.
    p_double: float = 0.6
    # Relative weight of mixed-spin doubles among all double excitations.
    mixed_double_weight: float = 1.0
    # Lower bound for |E_k - H_II| when evaluating the selection score.
    eps_denom: float = 1e-12

    verbose: bool = False
    orth_tol: float = 1e-12
