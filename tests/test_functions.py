"""Unit tests for registered CPU-bound functions."""

import time

import pytest

from edge_scheduler.functions import available, get, matrix_multiply, prime_search, hash_chain


class TestRegistry:
    def test_all_functions_registered(self):
        names = available()
        assert "matrix_multiply" in names
        assert "prime_search" in names
        assert "hash_chain" in names

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown function"):
            get("does_not_exist")

    def test_get_returns_callable(self):
        fn = get("matrix_multiply")
        assert callable(fn)


class TestMatrixMultiply:
    def test_returns_dict_with_checksum(self):
        result = matrix_multiply(n=10)
        assert isinstance(result, dict)
        assert "checksum" in result
        assert result["n"] == 10

    def test_takes_nonzero_time(self):
        start = time.perf_counter()
        matrix_multiply(n=50)
        elapsed = time.perf_counter() - start
        assert elapsed > 0

    def test_larger_n_takes_longer(self):
        start = time.perf_counter()
        matrix_multiply(n=20)
        small = time.perf_counter() - start

        start = time.perf_counter()
        matrix_multiply(n=60)
        large = time.perf_counter() - start

        assert large > small


class TestPrimeSearch:
    def test_known_prime_count(self):
        # There are 25 primes below 100.
        result = prime_search(n=100)
        assert result["count"] == 25
        assert result["largest"] == 97

    def test_returns_dict_with_count(self):
        result = prime_search(n=50)
        assert isinstance(result, dict)
        assert "count" in result
        assert "largest" in result

    def test_n1_returns_no_primes(self):
        result = prime_search(n=1)
        assert result["count"] == 0
        assert result["largest"] is None


class TestHashChain:
    def test_returns_hex_string(self):
        result = hash_chain(n=10)
        assert isinstance(result["final_hash"], str)
        assert len(result["final_hash"]) == 16   # first 8 bytes as hex

    def test_deterministic(self):
        r1 = hash_chain(n=100)
        r2 = hash_chain(n=100)
        assert r1["final_hash"] == r2["final_hash"]

    def test_different_n_gives_different_hash(self):
        r1 = hash_chain(n=100)
        r2 = hash_chain(n=101)
        assert r1["final_hash"] != r2["final_hash"]
