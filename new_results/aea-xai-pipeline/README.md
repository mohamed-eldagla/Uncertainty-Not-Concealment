# AEA-XAI condition pipeline (paper scope)

This directory contains only the artifacts used by the experiments in
`drafts/draft_20/paper.pdf`: C0/C2/C3/C4 construction, the split, and the
retrieval metadata used to build C4.

It does **not** contain MDI, AEA re-scan, RIVN, or calibration artifacts. Those
were legacy pipeline outputs and are not part of the paper.

## Files

| File | Purpose |
|---|---|
| `splits.csv` | 3,689 frontal reports with split, full text, MeSH/Problems label |
| `C1.csv` | matched: image + true report (n=738) |
| `C2.csv` | normal image + abnormal report (n=257) |
| `C3.csv` | abnormal image + normal report (n=481) |
| `C4.csv` | abnormal image + normal report among top-3 retrieved (n=234) |
| `retrieval.csv` | top-3 BiomedCLIP retrieval for every test image |
| `pipeline_audit.json` | counts, label purity, BiomedCLIP token audit |
| `build_pipeline_v2.py` | rebuilds all of the above from the IU dataset |

## Label rule

`normal` iff both `MeSH` and `Problems` contain the exact semicolon-separated
token `normal`. No substring matching (`abnormal` is not `normal`).
`No Indexing` is treated as abnormal.

## Images

Frontal projections only, resolved through `indiana_projections.csv`
(`projection == "Frontal"`). No glob-based image selection.

## Truncation

None. The condition CSVs store the full `findings + " " + impression` string.
BiomedCLIP token audit: train max = 242 tokens (limit 256), so the retrieval
corpus is not truncated. Test report text is not text-encoded.
