#!/usr/bin/env python3
"""Forced-choice re-parse: multiple distinct leading choices -> OTHER."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

CHOICES = {"SUPPORT", "SUPPORTS", "CONTRADICT", "CONTRADICTS", "CANNOT SAY",
           "CANNOT", "INCONCLUSIVE", "NO", "NONE", "NO INFORMATION", "NO OPINION"}


def norm_choice(x: str) -> str:
    if x.startswith("SUPPORT"):
        return "SUPPORTS"
    if x.startswith("CONTRADICT"):
        return "CONTRADICTS"
    return "OTHER"


def leading_choices(gen: str) -> list[str]:
    lines = [re.sub(r"[^A-Za-z ]", "", x).strip().upper() for x in str(gen or "").splitlines()]
    lines = [x for x in lines if x]
    out = []
    for x in lines[:4]:
        if x in CHOICES:
            out.append(norm_choice(x))
        else:
            break
    return out


def main() -> None:
    root = Path(__file__).resolve().parent / "new_results"
    p = root / "forced_choice_internvl2_paper" / "forced_choice_rows.csv"
    rows = list(csv.DictReader(open(p)))
    if "verdict_firstlabel" not in rows[0]:
        for r in rows:
            r["verdict_firstlabel"] = r["verdict"]
    for r in rows:
        lc = leading_choices(r["gen"])
        r["verdict"] = "OTHER" if len(set(lc)) > 1 else r["verdict_firstlabel"]
    fields = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    matched = [r for r in rows if r["tag"] == "matched"]
    mismatched = [r for r in rows if r["tag"] == "mismatched"]
    per = {}
    for cond in ["C2", "C3", "C4"]:
        sub = [r for r in mismatched if r["cond"] == cond]
        c = Counter(r["verdict"] for r in sub)
        per[cond] = {
            "counts": {k: int(v) for k, v in c.items()},
            "n": len(sub),
            "contradicts_rate": float(c.get("CONTRADICTS", 0) / max(len(sub), 1) * 100),
        }
    mc = Counter(r["verdict"] for r in matched)
    mmc = Counter(r["verdict"] for r in mismatched)
    results = {
        "model": "InternVL2-8B",
        "parser": "multiple leading choices -> OTHER; else first label/line; negation-aware",
        "mismatched_counts": {k: int(v) for k, v in mmc.items()},
        "mismatched_contradicts_rate": float(mmc.get("CONTRADICTS", 0) / max(len(mismatched), 1) * 100),
        "matched_counts": {k: int(v) for k, v in mc.items()},
        "matched_contradicts_rate": float(mc.get("CONTRADICTS", 0) / max(len(matched), 1) * 100),
        "per_condition": per,
        "n_mismatched": len(mismatched),
        "n_matched": len(matched),
    }
    json.dump(results, open(p.parent / "forced_choice_results.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
