"""
conftest.py — pytest configuration loaded automatically before any test runs.

This file adds the backend/ directory to Python's module search path so that
test files can import backend modules (search_tools, ai_generator, etc.)
directly by name, without needing relative import hacks in every test file.

pytest discovers and executes this file automatically — you never import it.
"""
import sys
import os

# Insert the backend/ directory (one level up from this tests/ folder)
# at the front of sys.path so `import search_tools` resolves correctly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
