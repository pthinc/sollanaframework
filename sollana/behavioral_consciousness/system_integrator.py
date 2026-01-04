"""Wrapper to expose Behavioral Consciousness system_integrator via sollana package."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Behavioral Consciousness"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system_integrator import *  # noqa: F401,F403
