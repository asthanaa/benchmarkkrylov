"""Compatibility wrapper for the advanced N2 threshold sweep driver."""

from advanced import n2_qp_threshold_sweep_plots as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})


if __name__ == "__main__":
    _impl.main()
