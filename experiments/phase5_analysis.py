"""Phase 5: generate the four analysis figures from the results CSVs.

Run: ``python experiments/phase5_analysis.py``
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, plotting

REQUIRED_CSVS = [config.LATENCY_CSV, config.MEMORY_CSV, config.ATTENTION_CSV]


def main() -> None:
    missing = [str(p) for p in REQUIRED_CSVS if not p.exists()]
    if missing:
        raise SystemExit(f"Missing results CSVs (run phases 2-4 first): {missing}")

    print("Generating figures ...")
    paths = plotting.generate_all()
    for p in paths:
        size_kb = Path(p).stat().st_size / 1024
        print(f"  wrote {p}  ({size_kb:.0f} KB)")
    print(f"\n{len(paths)} figures written to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
