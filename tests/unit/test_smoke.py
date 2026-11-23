"""Keeps CI meaningful from the first commit, before any component exists."""

import sys

import olist_mlops


def test_shared_package_imports():
    # Every container relies on `pip install -e .` making this importable.
    assert olist_mlops.__name__ == "olist_mlops"


def test_python_version_matches_pins():
    # The pins in requirements-dev.txt were checked together on Python 3.12 only.
    assert sys.version_info[:2] == (3, 12)
