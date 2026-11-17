#!/usr/bin/env python3
"""Re-parse saved forced-choice generations with a first-label parser."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

_NEG = re.compile(r"\b(NOT|NO|NEVER|WITHOUT|CANNOT|UNABLE|FAILS?|ABSENT|LACKS?)\b")


def is_negated(prefix: str) -> bool:
    return bool(_NEG.search(prefix)) or "N'T" in prefix


def forced_verdict(ans):
    a = str(ans or "").strip()
    if not a:
        return "OTHER", False
    first = re.match(r"\s*([A-Za-z]+)", a)
    if first:
        w = first.group(1).upper()
        if w.startswith("SUPPORT"):
            return "SUPPORTS", False
        if w.startswith("CONTRADICT"):
            return "CONTRADICTS", False
        if w.startswith("CANNOT") or w.startswith("INCONCLUSIVE"):
            return "OTHER", False
    m = re.search(r"\b(SUPPORTS?|CONTRADICTS?|CANNOT SAY|INCONCLUSIVE)\b", a, re.I)
    if m:
        w = m.group(1).upper()
        if w.startswith("CONTRADICT"):
            if is_negated(a[: m.start()].upper()):
                return "SUPPORTS", True
            return "CONTRADICTS", False
        if w.startswith("SUPPORT"):
            return "SUPPORTS", False
    return "OTHER", False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    p = Path(args.dir)
    rows_csv = p / "forced_choice_rows.csv"
    d = pd.read_csv(rows_csv)
    if "verdict_oldparser" not in d.columns:
        d["verdict_oldparser"] = d["verdict"]
    parsed = d["gen"].map(forced_verdict)
    d["verdict"] = [v for v, _ in parsed]
    d["negated"] = [n for _, n in parsed]
    d.to_csv(rows_csv, index=False)

    matched = d[d["tag"] == "matched"]
    mismatched = d[d["tag"] == "mismatched"]
    per = {}
    for cond in ["C2", "C3", "C4"]:
        sub = mismatched[mismatched["cond"] == cond]
        c = Counter(sub["verdict"])
        n = max(len(sub), 1)
        per[cond] = {
            "counts": {k: int(v) for k, v in c.items()},
            "n": int(len(sub)),
            "contradicts_rate": float(c.get("CONTRADICTS", 0) / n * 100),
        }
    mc = Counter(matched["verdict"])
    mmc = Counter(mismatched["verdict"])
    results = {
        "model": "InternVL2-8B",
        "parser": "first label / first line, brackets optional",
        "mismatched_counts": {k: int(v) for k, v in mmc.items()},
        "mismatched_contradicts_rate": float(mmc.get("CONTRADICTS", 0) / max(len(mismatched), 1) * 100),
        "matched_counts": {k: int(v) for k, v in mc.items()},
        "matched_contradicts_rate": float(mc.get("CONTRADICTS", 0) / max(len(matched), 1) * 100),
        "per_condition": per,
        "n_mismatched": int(len(mismatched)),
        "n_matched": int(len(matched)),
    }
    json.dump(results, open(p / "forced_choice_results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
