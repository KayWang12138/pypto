"""Test utilities and path setup for pypto tests.

This module automatically sets up the Python path so that all test files
can directly import pypto without manual path manipulation.

Usage in test files:
    import tests  # noqa: F401 - triggers path setup
    import pypto
"""
import sys
from pathlib import Path

# Add parent directory (python/) to sys.path so pypto can be imported
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

