import re
from pathlib import Path
import numpy as np
from p4reader import P4History


def collect_last_history_values(base_dir=".", traces=()):
    base_dir = Path(base_dir)

    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and re.fullmatch(r"run\d+", d.name)],
        key=lambda d: int(d.name[3:]),
    )

    vals = {trace: [] for trace in traces}

    for run_dir in run_dirs:
        hist_file = run_dir / "history.p4"
        if not hist_file.exists():
            continue

        hist = P4History(str(hist_file))

        for trace in traces:
            vals[trace].append(hist[trace][-1])

    return {trace: np.array(vals[trace]) for trace in traces}
