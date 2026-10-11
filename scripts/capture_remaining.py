"""Compatibility entrypoint: capture current UI through the browser smoke test."""
from pathlib import Path
import subprocess
import sys

raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("browser_smoke.py")), "--screenshots"]))
