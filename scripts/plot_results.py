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
    # one line per sweep: strip the query-time knob (nprobe / ef) from params;
    # flat pq has no such knob, so its line runs over m instead
    p = re.sub(r"\s*(nprobe|ef)=\d+", "", row["params"])
    if row["index"] == "pq":
        p = re.sub(r"\s*m=\d+", "", p)
    return f'{row["index"]} {p.strip()}'.strip()


def main() -> None:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.csv")):
        with path.open() as f:
            for r in csv.DictReader(f):
                groups[series_key(r)].append((float(r["qps"]), float(r["recall"])))

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = "osD^v<>Pph*"
    for i, (name, pts) in enumerate(sorted(groups.items())):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=markers[i % len(markers)], ms=5, label=name, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("queries / second (log)")
    ax.set_ylabel("recall@10")
    ax.set_ylim(0.15, 1.02)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("recall vs throughput, single thread")
    out = RESULTS / "recall_vs_qps.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
