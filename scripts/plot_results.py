"""Plot recall@k against QPS for every result row, one line per index/nlist.

    uv run scripts/plot_results.py   # writes results/recall_vs_qps.png
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parents[1] / "results"


def series_key(row: dict) -> str:
    # group sweeps: strip the query-time knob (nprobe / ef_search) from params
    p = re.sub(r"\s*(nprobe|ef)=\d+", "", row["params"]).strip()
    return f'{row["index"]} {p}'.strip()


def main() -> None:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                groups[series_key(r)].append((float(r["qps"]), float(r["recall"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pts in groups.items():
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("queries / second (log)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("recall vs throughput, single thread")
    out = RESULTS / "recall_vs_qps.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
