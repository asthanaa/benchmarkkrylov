"""Compatibility wrapper for shared Quantum Package helper utilities."""

from _shared import qp_ccik_common as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
