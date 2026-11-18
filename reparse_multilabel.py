#!/usr/bin/env python3
"""Re-parse saved three-way verdicts with multi-label -> INVALID."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

VID = re.compile(r"VISUAL\s+EVIDENCE\s*:?\s*\[?\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)", re.I)
BARE = re.compile(r"^\s*\[?\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\b", re.I)
ALL3 = ("SUPPORTS", "CONTRADICTS", "INCONCLUSIVE")


def verdict(gen: str) -> str:
    t = str(gen or "").strip().upper()
    first_line = t.splitlines()[0] if t else ""
    # template echo, e.g. [SUPPORTS/CONTRADICTS/INCONCLUSIVE]
    if all(x in first_line for x in ALL3) and ("/" in first_line or "|" in first_line):
        return "INVALID"
    labels = [m.group(1).upper() for m in VID.finditer(t)]
    if len(set(labels)) > 1:
        return "INVALID"
    if labels:
        return labels[0]
    m = BARE.match(t)
    return m.group(1).upper() if m else "INVALID"


def reparse_behavioral(root: Path) -> None:
    for name in ["internvl2_sft_output_paper", "qwen2vl_output_paper", "llava7b_paper", "llava13b_paper"]:
        p = root / name / "aea_verdicts.csv"
        rows = list(csv.DictReader(open(p)))
        if not rows:
            continue
        old_col = "verdict_oldparser"
        if old_col not in rows[0]:
            for r in rows:
                r[old_col] = r["verdict"]
        for r in rows:
            r["verdict"] = verdict(r.get("gen", ""))
        fields = list(rows[0].keys())
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        # summary
        per = {}
        for cond in ["C2", "C3", "C4"]:
            sub = [r for r in rows if r["cond"] == cond]
            c = Counter(r["verdict"] for r in sub)
            n = max(len(sub), 1)
            per[cond] = {
                "n": len(sub),
                **{k: int(c.get(k, 0)) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
                **{f"{k}_pct": float(c.get(k, 0) / n * 100) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
            }
        sp = root / name / "summary.json"
        if sp.exists():
            s = json.load(open(sp))
            s["verdicts"] = {k: int(v) for k, v in Counter(r["verdict"] for r in rows).items()}
            s["per_condition"] = per
            s["verdict_parser"] = "first VISUAL EVIDENCE label; multi-label outputs INVALID"
            json.dump(s, open(sp, "w"), indent=2)
        print(name, "updated", len(rows), "rows")


def reparse_caas(root: Path) -> None:
    p = root / "caas_internvl2_paper" / "steering_generations_raw.csv"
    rows = list(csv.DictReader(open(p)))
    if "robust_firstlabel" not in rows[0]:
        for r in rows:
            r["robust_firstlabel"] = r["robust"]
    for r in rows:
        r["robust"] = verdict(r.get("gen", ""))
    fields = list(rows[0].keys())
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    jp = root / "caas_internvl2_paper" / "caas_results.json"
    s = json.load(open(jp))
    sweep = defaultdict(list)
    fp = []
    for r in rows:
        if r["tag"] == "sweep":
            sweep[r["alpha"]].append(r)
        elif r["tag"] == "fp":
            fp.append(r)
    new_sweep = {}
    for alpha in sorted(sweep, key=float):
        sub = sweep[alpha]
        rc = Counter(r["robust"] for r in sub)
        nc = Counter(r["naive"] for r in sub)
        n = len(sub)
        new_sweep[alpha] = {
            "naive": dict(nc),
            "robust": dict(rc),
            "naive_cont": float(nc.get("CONTRADICTS", 0) / n * 100),
            "robust_cont": float(rc.get("CONTRADICTS", 0) / n * 100),
        }
    fp_alpha = str(s.get("fp_alpha", 20.0))
    fsub = [r for r in fp if r["alpha"] == fp_alpha]
    fc = Counter(r["robust"] for r in fsub)
    fn = Counter(r["naive"] for r in fsub)
    s["sweep"] = new_sweep
    s["false_positive_naive_counts"] = dict(fn)
    s["false_positive_robust_counts"] = dict(fc)
    s["false_positive_robust_contradicts_rate"] = float(fc.get("CONTRADICTS", 0) / max(len(fsub), 1) * 100)
    s["verdict_parser"] = "first VISUAL EVIDENCE label; multi-label outputs INVALID"
    json.dump(s, open(jp, "w"), indent=2)
    print("caas updated", len(rows), "rows; fp n", len(fsub))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "new_results"
    reparse_behavioral(root)
    reparse_caas(root)
