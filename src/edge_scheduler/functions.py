"""Registered CPU-bound functions that nodes can execute.

Each function accepts a plain dict of args and returns a JSON-serialisable
result. They are intentionally CPU-heavy so they create real measurable load.
"""

from __future__ import annotations

import hashlib
import random


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, callable] = {}


def register(name: str):
    """Decorator to register a function by name."""
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get(name: str):
    """Look up a registered function by name. Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown function '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Registered functions
# ---------------------------------------------------------------------------

@register("matrix_multiply")
def matrix_multiply(n: int = 100) -> dict:
    """Multiply two N×N matrices of random floats.

    Args:
        n: Matrix dimension. n=100 is fast (~1ms), n=400 is heavier (~200ms).
    """
    # Pure Python — intentionally slow to generate real CPU load.
    a = [[random.random() for _ in range(n)] for _ in range(n)]
    b = [[random.random() for _ in range(n)] for _ in range(n)]
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            for j in range(n):
                result[i][j] += a[i][k] * b[k][j]
    # Return the sum of the first row as a sanity-check value.
    return {"n": n, "checksum": round(sum(result[0]), 4)}


@register("prime_search")
def prime_search(n: int = 10_000) -> dict:
    """Find all prime numbers up to N using the Sieve of Eratosthenes.

    Args:
        n: Upper bound. n=10_000 is fast, n=500_000 is heavier.
    """
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i, v in enumerate(sieve) if v]
    return {"n": n, "count": len(primes), "largest": primes[-1] if primes else None}


@register("hash_chain")
def hash_chain(n: int = 50_000) -> dict:
    """Compute SHA-256 hashed N times in a chain (each hash feeds into the next).

    Args:
        n: Number of hash iterations. n=50_000 is moderate, n=500_000 is heavier.
    """
    data = b"edge-scheduler-seed"
    for _ in range(n):
        data = hashlib.sha256(data).digest()
    return {"n": n, "final_hash": data.hex()[:16]}  # first 8 bytes as hex
