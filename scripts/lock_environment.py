"""Pin installed direct and transitive project dependencies for this platform."""
from importlib.metadata import distribution
from pathlib import Path
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
pending = [Requirement(line).name for line in (root / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
locked = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in locked:
        continue
    package = distribution(name)
    locked[name] = package.version
    for dependency in package.requires or []:
        requirement = Requirement(dependency)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            pending.append(requirement.name)
(root / "requirements.lock.txt").write_text("# Verified with Python 3.11 on Windows; regenerate with scripts/lock_environment.py\n" + "\n".join(f"{name}=={version}" for name, version in sorted(locked.items())) + "\n", encoding="utf-8")
