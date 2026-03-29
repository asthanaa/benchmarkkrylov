"""Compatibility wrapper for the advanced Quantum Package bridge tool."""

from advanced import qp_ccik_bridge as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})


if __name__ == "__main__":
    _impl.main()
