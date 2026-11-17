# Uncertainty, Not Concealment

Code for *Uncertainty, Not Concealment: A Mechanistic Reappraisal of Sycophancy in Vision-Language Models*.

The code evaluates instruction-tuned vision-language models on frontal chest X-rays with matched and mismatched image-report inputs. It covers behavioral prompting, CLS probes, activation steering, causal patching, two-way forced choice, and cross-modal attention probes.

## Code layout

- `qwen2vl/` — Qwen2-VL evaluation.
- `internvl2_sft/` — InternVL2 evaluation and mechanistic experiments.
- `llava_behavioral_cls.py` — LLaVA-1.5 behavioral and CLS evaluation.
- `reparse_verdicts.py`, `reparse_forced_choice.py` — response parsers.
- `new_results/` — experiment outputs.

Models are loaded in 8-bit. Inputs use frontal images and full report text.
