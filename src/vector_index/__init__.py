"""Vector indexes from scratch, plus a benchmark harness."""

from .brute import BruteForceIndex
from .index import Index

__all__ = ["Index", "BruteForceIndex"]
