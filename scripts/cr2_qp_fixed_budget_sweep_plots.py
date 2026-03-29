"""Compatibility wrapper for the advanced Cr2 fixed-budget sweep driver."""

from advanced import cr2_qp_fixed_budget_sweep_plots as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})


if __name__ == "__main__":
    _impl.main()
