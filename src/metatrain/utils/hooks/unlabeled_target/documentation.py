"""
Identity
========

This hook just passes the inputs to the outputs without any modification.

For someone who wants to understand how hooks work or wants to implement
their own hook, this is the best point to start, as it is the simplest
hook possible.

It is also very useful for testing and debugging.
"""

from typing import Optional

from typing_extensions import TypedDict


class Hypers(TypedDict):
    """
    Hyperparameters for the identity hook.
    """

    targets: dict[str, dict]