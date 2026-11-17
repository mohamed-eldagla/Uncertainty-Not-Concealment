#!/usr/bin/env python3
"""Re-parse saved generations with the paper's robust first-verdict rule."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def robust_verdict(text) -> str:
    t = str(text or "").strip().upper()
    first_line = t.splitlines()[0] if t else ""
    # A template echo is not a choice: VISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]
    if (
        all(x in first_line for x in ("SUPPORTS", "CONTRADICTS", "INCONCLUSIVE"))
        and ("/" in first_line or "|" in first_line)
    ):
        return "INVALID"
    m = re.search(r"VISUAL EVIDENCE\s*:?\s*\[?\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)", t)
    if m:
        return m.group(1)
    m = re.match(r"\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\b", t)
    if m:
        return m.group(1)
    return "INVALID"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    p = Path(args.dir)
    csv = p / "aea_verdicts.csv"
    d = pd.read_csv(csv)
    if "gen" not in d.columns:
        raise SystemExit(f"no gen column in {csv}")
    if "verdict_oldparser" not in d.columns:
        d["verdict_oldparser"] = d["verdict"]
    d["verdict"] = d["gen"].map(robust_verdict)
    d.to_csv(csv, index=False)

    per = {}
    for cond, sub in d.groupby("cond"):
        n = max(len(sub), 1)
        per[str(cond)] = {
            "n": int(len(sub)),
            **{k: int((sub["verdict"] == k).sum()) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
            **{f"{k}_pct": float((sub["verdict"] == k).mean() * 100) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
        }
    summary_path = p / "summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        summary["verdicts"] = {k: int(v) for k, v in d["verdict"].value_counts().items()}
        summary["per_condition"] = per
        summary["verdict_parser"] = "first VISUAL EVIDENCE label, brackets optional"
        json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"updated {csv}")
    print(d["verdict"].value_counts().to_string())
    print(pd.DataFrame(per).T.to_string())


if __name__ == "__main__":
    main()
