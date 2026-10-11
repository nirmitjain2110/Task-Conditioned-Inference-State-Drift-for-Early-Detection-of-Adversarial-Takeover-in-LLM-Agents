"""
Backwards-compatible wrapper calling scripts/07_visualize.py.
"""
import os
import sys
import runpy

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(__file__), "07_visualize.py")
    runpy.run_path(script_path, run_name="__main__")
