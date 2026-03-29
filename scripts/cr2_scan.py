"""Compatibility wrapper for the advanced Cr2 scan driver."""

from advanced.cr2_scan import *  # noqa: F401,F403


if __name__ == "__main__":
    from advanced.cr2_scan import main

    main()
