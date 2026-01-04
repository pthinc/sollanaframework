"""Wrapper to expose backends.trainer_bridge via sollana package."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backends"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from trainer_bridge import *  # noqa: F401,F403
