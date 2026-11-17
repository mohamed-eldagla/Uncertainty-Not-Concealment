#!/usr/bin/env python3
"""
Regenerate the AEA-XAI MDI / AEA outputs on the corrected conditions.

No input truncation:
  - baseline prompt uses the full retrieved report
  - AEA prompt uses the full retrieved report
  - rationale is stored in full (no character cap)

Outputs (in --pipeline-dir):
  mdi_base.csv, trajs_base.pkl,
  aea_verdicts.csv, trajs_aea.pkl,
  delta_stats.json, fig_mdi_trajectory.png, fig_mdi_dist.png,
  mdi_run_status.json
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)

SEED = 42

BASELINE_TMPL = (
    "USER: <image>\nRetrieved record: {report}\n"
    "Diagnosis based on image and record?\nASSISTANT:"
)
AEA_TMPL = (
    "USER: <image>\nRetrieved record suggests: {finding}.\n"
    "Does this X-ray SUPPORT, CONTRADICT, or is INCONCLUSIVE about '{finding}'?\n"
    "Check: (a) lung opacification (b) costophrenic angles "
    "(c) cardiac silhouette (d) pleural space.\n"
    "Reply ONLY:\nVISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]\n"
    "RATIONALE: [one sentence]\nASSISTANT:"
)


def naive_verdict(text: str) -> str:
    t = (text or "").upper()
    if "CONTRADICTS" in t:
        return "CONTRADICTS"
    if "SUPPORTS" in t:
        return "SUPPORTS"
    return "INCONCLUSIVE"


def robust_verdict(text: str) -> str:
    t = text or ""
    m = re.search(r"\[\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\s*\]", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.match(r"\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\b", t, re.I)
    if m:
        return m.group(1).upper()
    return "INVALID"


def find_layers(model):
    lm = getattr(model, "language_model", None)
    if lm is not None:
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    inner = getattr(model, "model", None)
    if inner is not None:
        if hasattr(inner, "layers"):
            return inner.layers
        lm = getattr(inner, "language_model", None)
        if lm is not None:
            if hasattr(lm, "layers"):
                return lm.layers
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
    raise RuntimeError("could not locate transformer layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-dir", default="/kaggle/working/aea-xai-pipeline-v2")
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test limit per condition")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    P = Path(args.pipeline_dir)
    OUT = Path(args.out_dir or P)
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} pipeline={P} out={OUT}", flush=True)

    splits = pd.read_csv(P / "splits.csv")
    image_by_uid = splits.set_index("uid")["image_path"].to_dict()
    rivn_df = pd.read_csv(P / "rivn.csv")
    rivn = {r.uid: int(r.rivn) for r in rivn_df.itertuples()}

    print("loading LLaVA...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model_id)
    llava = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    llava.eval()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    layers = find_layers(llava)
    print(f"layers={len(layers)} hf_device_map={getattr(llava, 'hf_device_map', None)}", flush=True)
    out_emb = llava.get_output_embeddings()
    W_U = out_emb.weight.float()
    img_tok = proc.tokenizer.convert_tokens_to_ids("<image>")
    print(f"image_token_id={img_tok}", flush=True)

    def mdi_pass(uid: str, report: str, mode: str = "baseline"):
        path = image_by_uid.get(uid)
        if not path or not Path(path).is_file():
            return None, None, None
        img = Image.open(path).convert("RGB")
        prompt = (
            BASELINE_TMPL.format(report=report)
            if mode == "baseline"
            else AEA_TMPL.format(finding=report)
        )
        inp = proc(text=prompt, images=img, return_tensors="pt").to(llava.device)
        ids = inp["input_ids"][0]
        vis_pos = (ids == img_tok).nonzero(as_tuple=True)[0]
        txt_pos = (ids != img_tok).nonzero(as_tuple=True)[0]
        if len(vis_pos) == 0:
            mid = len(ids) // 2
            vis_pos = torch.arange(0, mid, device=ids.device)
            txt_pos = torch.arange(mid, len(ids), device=ids.device)

        hs_store = {}

        def get_hook(layer_idx):
            def hook(module, inputs, output):
                val = output[0] if isinstance(output, tuple) else output
                hs_store[layer_idx] = val.detach().float()

            return hook

        hooks = [layer.register_forward_hook(get_hook(i)) for i, layer in enumerate(layers)]
        with torch.no_grad():
            llava(**inp)
        for hook in hooks:
            hook.remove()

        traj = []
        for layer_idx in sorted(hs_store):
            hs = hs_store[layer_idx][0]

            def entropy(pos):
                pos_local = pos.to(hs.device)
                logits = hs[pos_local].mean(0) @ W_U.to(hs.device).t()
                p = torch.softmax(logits, dim=-1)
                return -(p * (p + 1e-9).log()).sum().item()

            traj.append(entropy(txt_pos) / (entropy(vis_pos) + 1e-9))

        mdi = traj[-2] if len(traj) >= 2 else traj[-1]
        gen_text = None
        if mode == "aea":
            with torch.no_grad():
                g = llava.generate(
                    **inp, max_new_tokens=args.max_new_tokens, do_sample=False
                )
            gen_text = proc.decode(
                g[0][inp["input_ids"].shape[1] :], skip_special_tokens=True
            )
        torch.cuda.empty_cache()
        return float(mdi), traj, gen_text

    # ---------------------------------------------------- baseline MDI
    mdi_rows, trajs_base = [], {}
    for cond in ["C1", "C2", "C3", "C4"]:
        pairs = pd.read_csv(P / f"{cond}.csv").dropna(subset=["ret_report"])
        if args.limit:
            pairs = pairs.head(args.limit)
        print(f"baseline {cond}: {len(pairs)} pairs", flush=True)
        for idx, row in pairs.iterrows():
            if idx % 50 == 0:
                print(f"  {cond} {idx}/{len(pairs)}", flush=True)
            mdi_val, traj, _ = mdi_pass(row["uid"], row["ret_report"], "baseline")
            if mdi_val is None:
                continue
            mdi_rows.append(
                {
                    "uid": row["uid"],
                    "cond": cond,
                    "mdi_base": float(mdi_val),
                    "rivn": int(rivn.get(row["uid"], 0)),
                }
            )
            trajs_base[f"{cond}_{row['uid']}"] = traj
    mdi_df = pd.DataFrame(mdi_rows)
    mdi_df.to_csv(OUT / "mdi_base.csv", index=False)
    pickle.dump(trajs_base, open(OUT / "trajs_base.pkl", "wb"))
    print("baseline cond means:\n", mdi_df.groupby("cond")["mdi_base"].mean().round(3), flush=True)

    # ---------------------------------------------------- AEA re-scan
    rivn_uids = {u for u, v in rivn.items() if v == 1}
    aea_rows, trajs_aea = [], {}
    for cond in ["C2", "C3", "C4"]:
        pairs = pd.read_csv(P / f"{cond}.csv")
        flagged = pairs[pairs["uid"].isin(rivn_uids)]
        if args.limit:
            flagged = flagged.head(args.limit)
        print(f"AEA {cond}: {len(flagged)} flagged", flush=True)
        for idx, row in flagged.iterrows():
            if idx % 20 == 0:
                print(f"  {cond} {idx}/{len(flagged)}", flush=True)
            mdi_p, traj, gen = mdi_pass(row["uid"], row["ret_report"], "aea")
            if mdi_p is None:
                continue
            base = mdi_df[
                (mdi_df["uid"] == row["uid"]) & (mdi_df["cond"] == cond)
            ]["mdi_base"]
            if base.empty:
                continue
            delta = float(base.values[0]) - float(mdi_p)
            rat_i = (gen or "").lower().find("rationale:")
            rationale = (gen[rat_i + 10 :].strip() if rat_i >= 0 else "")
            aea_rows.append(
                {
                    "uid": row["uid"],
                    "cond": cond,
                    "mdi_base": float(base.values[0]),
                    "mdi_prime": float(mdi_p),
                    "delta": delta,
                    "verdict": naive_verdict(gen),
                    "verdict_robust": robust_verdict(gen),
                    "rationale": rationale,
                    "gen": gen,
                }
            )
            trajs_aea[f"{cond}_{row['uid']}"] = traj
    aea_df = pd.DataFrame(aea_rows)
    aea_df.to_csv(OUT / "aea_verdicts.csv", index=False)
    pickle.dump(trajs_aea, open(OUT / "trajs_aea.pkl", "wb"))

    deltas = aea_df["delta"].dropna().values
    if len(deltas) > 10:
        W, p = wilcoxon(deltas)
        stats = {"mean_delta": float(deltas.mean()), "W": float(W), "p": float(p)}
        json.dump(stats, open(OUT / "delta_stats.json", "w"))
        print(f"delta-MDI={deltas.mean():.3f} W={W:.1f} p={p:.4f}", flush=True)

    # ---------------------------------------------------- figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def mean_traj(store, cond):
        ts = [v for k, v in store.items() if k.startswith(cond) and v]
        if not ts:
            return None
        n = min(len(t) for t in ts)
        return np.array([t[:n] for t in ts]).mean(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    for cond, col in [("C1", "steelblue"), ("C2", "tomato")]:
        tb = mean_traj(trajs_base, cond)
        ta = mean_traj(trajs_aea, cond)
        if tb is not None:
            ax.plot(tb, "--", color=col, label=f"{cond} Naive RAG", lw=1.8)
        if ta is not None:
            ax.plot(ta, "-", color=col, label=f"{cond} AEA", lw=1.8)
    ax.axhline(1.0, color="k", lw=1, ls=":", label="MDI=1")
    ax.set_xlabel("Layer")
    ax.set_ylabel("MDI")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("MDI Layer Trajectory: Naive RAG vs AEA")
    plt.tight_layout()
    plt.savefig(OUT / "fig_mdi_trajectory.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    data = [
        mdi_df[mdi_df["cond"] == c]["mdi_base"].dropna().values
        for c in ["C1", "C2", "C3", "C4"]
    ]
    vp = ax.violinplot(data, positions=[1, 2, 3, 4], showmedians=True)
    for body in vp["bodies"]:
        body.set_alpha(0.7)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["C1", "C2", "C3", "C4"])
    ax.set_ylabel("MDI")
    ax.set_title("MDI Distribution by Condition")
    plt.tight_layout()
    plt.savefig(OUT / "fig_mdi_dist.png", dpi=300)
    plt.close()

    status = {
        "model_id": args.model_id,
        "pipeline_dir": str(P),
        "n_mdi": int(len(mdi_df)),
        "n_aea": int(len(aea_df)),
        "mdi_by_cond": {
            k: int(v) for k, v in mdi_df["cond"].value_counts().items()
        },
        "aea_by_cond": {
            k: int(v) for k, v in aea_df["cond"].value_counts().items()
        },
        "rivn_flagged": int(sum(rivn.values())),
        "rivn_n": int(len(rivn)),
        "max_new_tokens": args.max_new_tokens,
        "input_truncation": "none",
        "rationale_truncation": "none",
        "verdicts_naive": {
            k: int(v) for k, v in aea_df["verdict"].value_counts().items()
        },
        "verdicts_robust": {
            k: int(v) for k, v in aea_df["verdict_robust"].value_counts().items()
        },
        "delta_stats": (
            json.load(open(OUT / "delta_stats.json"))
            if (OUT / "delta_stats.json").exists()
            else None
        ),
    }
    json.dump(status, open(OUT / "mdi_run_status.json", "w"), indent=2)
    print(json.dumps(status, indent=2), flush=True)
    del llava
    gc.collect()
    torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
