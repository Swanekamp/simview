from __future__ import annotations
from pathlib import Path
import re

# ----------------------------
# Utilities
# ----------------------------

def _ns_tag(t_ns: float) -> str:
    # 10ns, 05ns, 2.5ns -> consistent filenames
    if abs(t_ns - round(t_ns)) < 1e-6:
        return f"{int(round(t_ns)):02d}ns"
    return f"{t_ns:05.1f}ns".replace(".", "p")


def _sorted_flds_files(run_dir: str | Path, pattern="flds*.p4"):
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob(pattern))
    # If you prefer numeric sort by flds####:
    def key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 10**9
    return sorted(files, key=key)

