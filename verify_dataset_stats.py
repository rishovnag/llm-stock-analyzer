# Verify the dataset-level statistics reported in the paper (Table 1:
# "Incidence of positive jump days") directly from the CSVs in data/.
# This is a fast, API-free integrity check that the released data matches the
# numbers in the manuscript.
#
# Usage:  python verify_dataset_stats.py

from pathlib import Path
import pandas as pd

DIRECTION_FILES = {
    "NIFTY50": "NIFTY50_2007_2025_with_Direction.csv",
    "BANKNIFTY": "NIFTYBANK_2007_2025_Daily_with_Direction.csv",
    "NIFTYIT": "NIFTYIT_2007_2025_Daily_with_Direction.csv",
}

# Values reported in the paper's Table 1: (days 1-2%, days 2-3%, days >=3%)
PAPER_TABLE1 = {
    "NIFTY50": (486, 115, 77),
    "BANKNIFTY": (523, 219, 173),
    "NIFTYIT": (542, 184, 126),
}


def main(data_dir="data"):
    data_dir = Path(data_dir)
    all_ok = True
    print(f"{'Index':10s} {'computed (1-2,2-3,>=3)':28s} {'paper':18s} match")
    print("-" * 70)
    for index, fname in DIRECTION_FILES.items():
        df = pd.read_csv(data_dir / fname, thousands=",")
        price = pd.to_numeric(df["Adj_Close"], errors="coerce")
        pct = 100 * (price / price.shift(1) - 1)
        b12 = int(((pct >= 1) & (pct < 2)).sum())
        b23 = int(((pct >= 2) & (pct < 3)).sum())
        b3 = int((pct >= 3).sum())
        computed = (b12, b23, b3)
        expected = PAPER_TABLE1[index]
        ok = computed == expected
        all_ok &= ok
        print(f"{index:10s} {str(computed):28s} {str(expected):18s} "
              f"{'OK' if ok else 'MISMATCH'}")
    print("-" * 70)
    print("ALL MATCH ✅" if all_ok else "MISMATCHES FOUND ❌")
    return all_ok


if __name__ == "__main__":
    main()
