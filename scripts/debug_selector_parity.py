"""Compatibility wrapper for the advanced selector-parity helper."""

from _shared import debug_selector_parity as _impl

globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
