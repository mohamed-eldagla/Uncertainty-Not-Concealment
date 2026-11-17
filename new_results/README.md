# Corrected results (new)

All results here use the corrected condition pipeline:

- C1 738, C2 257, C3 481, C4 234
- exact MeSH/Problems labels
- frontal-only images
- full report text, no input truncation

## Result directories

| Directory | Contents | Status |
|---|---|---|
| `internvl2_sft_output_paper/` | InternVL2-8B behavioral verdicts, CLS features, summary | done |
| `qwen2vl_output_paper/` | Qwen2-VL behavioral verdicts, CLS features, summary | done |
| `llava7b_paper/` | LLaVA-7B behavioral verdicts, CLS features, CLIP control, summary | done |
| `llava13b_paper/` | LLaVA-13B behavioral verdicts, CLS features, CLIP control, summary | done |
| `forced_choice_internvl2_paper/` | Forced-choice rows and results | done |
| `w2_probes_paper/` | Attention, entropy, and CLIP-cosine probes | done |
| `caas_internvl2_paper/` | CAAS steering sweep and direction diagnostics | done |
| `causal_patching_paper/` | Visual-token causal-patching results | done |
| `aea-xai-pipeline/` | Corrected condition CSVs used by every run | done |
| `pipeline_v2_backup/` | Staging duplicate of the corrected condition pipeline | backup |
| `logs/` | Run logs | — |

The canonical new scripts are at the `final_codebase` root
(`internvl2_sft/`, `qwen2vl/`, `llava_behavioral_cls.py`, etc.), not duplicated
inside `new_results/`.

W3 bootstrap/permutation is excluded: the analysis used an ungrouped
train/test split, so matched and mismatched versions of the same image could
cross the split. The grouped-CV CLS AUC in Table 2 is unaffected. The W3
artifact is kept under `../old/legacy_results/w3_bootstrap_ungrouped_leaky/`.

## Key corrected numbers

- InternVL2-8B CLS AUC: 0.638 ± 0.031
- Qwen2-VL CLS AUC: 0.654 ± 0.043
- LLaVA-7B CLS AUC: 0.650 ± 0.019; CLIP AUC 0.531
- LLaVA-13B CLS AUC: 0.647 ± 0.013; CLIP AUC 0.531
- Attention AUC 0.438 (p=0.033); CLIP cosine AUC 0.595 (p=0.001); entropy AUC 0.466 (p=0.242)
- Forced choice CONTRADICTS: C0 1.3%, C2 3.0%, C3 0%, C4 0%
- CAAS robust CONTRADICTS: 0% at every α; false-positive 0%
- Causal patching layer 20: −0.0102 / +0.0151; layer 30: 0; no-op: 0
