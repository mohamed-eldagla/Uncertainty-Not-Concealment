# Regenerated AEA-XAI pipeline (staging)

This directory is the new condition/retrieval/calibration pipeline. It replaces
the old heuristic-labelled `corrected-aea-xai-pipeline` once the MDI/AEA outputs
are regenerated.

## What changed

- Frontal projections only. Image paths now come from
  `indiana_projections.csv` (`projection == "Frontal"`), not from a glob. The old
  glob could select the lateral image first (e.g., uid 1 -> `1_IM-0001-3001`
  instead of the frontal `1_IM-0001-4001`).
- Exact labels: `normal` iff both `MeSH` and `Problems` contain the exact
  semicolon-separated token `normal`. No substring matching, so `abnormal`
  cannot match. `No Indexing` is treated as abnormal (not normal), which
  preserves the 738-row test set and C0.
- No input truncation anywhere in this pipeline. The condition CSVs contain the
  full `findings + " " + impression` text.
- BiomedCLIP token audit: train max = 242 tokens (limit 256); only two test
  reports exceed 256 (292 and 264 tokens) and test report text is not encoded.
  No BiomedCLIP truncation occurs.

## Sizes

| Split | n | normal | abnormal |
|---|---:|---:|---:|
| train | 2582 | 956 | 1626 |
| val | 369 | 130 | 239 |
| test | 738 | 257 | 481 |

| Condition | n | Intended pair |
|---|---:|---|
| C1 | 738 | matched: image + true report |
| C2 | 257 | normal image + abnormal report |
| C3 | 481 | abnormal image + normal report |
| C4 | 234 | abnormal image + normal report among top-3 retrieved |

Condition purity is 100% under MeSH/Problems: C2 is 257/257 normal-image +
abnormal-report, C3 is 481/481 abnormal-image + normal-report, and C4 is
234/234 abnormal-image + normal-report.

C4 is 234 rather than the previously estimated 132 because the old retrieval
used a glob that sometimes selected lateral images. With strict frontal
selection, more abnormal images have a true normal report in the top 3. C4 uses
56 distinct reports; the most reused report is used 22/234 times (9.4%).

## RIVN caveat

CMCG calibration on the corrected labels: tau = 0.52, F1 = 0.617, AUC = 0.487.
It flags 732/738 test images. The RIVN gate is therefore essentially
non-selective; this is a property of the similarity score under correct labels,
not a label bug. Flagged for the revision.

## Pending

`mdi_base.csv`, `trajs_base.pkl`, `trajs_aea.pkl`, `aea_verdicts.csv`,
`delta_stats.json`, and the MDI figures require LLaVA-1.5-7B 8-bit. The Kaggle
ngrok tunnel dropped during the LLaVA checkpoint load; rerun
`run_mdi_aea_v2.py` when the tunnel is back.

Scripts:

- `build_pipeline_v2.py` — condition/retrieval/RIVN builder.
- `run_mdi_aea_v2.py` — baseline MDI + AEA re-scan with full-text prompts and
  full rationale storage.
