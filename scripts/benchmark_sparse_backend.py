"""Compatibility wrapper for the advanced sparse-backend benchmark driver."""

from advanced.benchmark_sparse_backend import *  # noqa: F401,F403


if __name__ == "__main__":
    from advanced.benchmark_sparse_backend import main

    main()
