#!/usr/bin/env python3
"""
LLaVA-1.5 behavioral verdicts + CLS probe + CLIP control (paper scope).

No MDI, no trajectories, no C1/AEA-delta. This is the paper's LLaVA
behavioral + CLS experiment only:

  - C0: baseline CLS features
  - C2/C3: baseline CLS features + three-way behavioral verdicts
  - C4: three-way behavioral verdicts
  - CLIP cosine control for C0/C2/C3/C4
  - nested grouped CLS probe (C0 vs C2/C3)
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

BT = (
    "USER: <image>\nRetrieved record: {report}\n"
    "Diagnosis based on image and record?\nASSISTANT:"
)
AT = (
    "USER: <image>\nRetrieved record suggests: {finding}.\n"
    "Does this X-ray SUPPORT, CONTRADICT, or is INCONCLUSIVE about '{finding}'?\n"
    "Check: (a) lung opacification (b) costophrenic angles "
    "(c) cardiac silhouette (d) pleural space.\n"
    "Reply ONLY:\nVISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]\n"
    "RATIONALE: [one sentence]\nASSISTANT:"
)


def robust_verdict(text: str) -> str:
    t = (text or "").strip().upper()
    first_line = t.splitlines()[0] if t else ""
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
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--out-dir", default="/kaggle/working/llava7b_paper")
    ap.add_argument("--pipeline-dir", default="/kaggle/working/aea-xai-pipeline")
    ap.add_argument("--iu-dir", default="/kaggle/input/datasets/raddar/chest-xrays-indiana-university")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUT = Path(args.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True, parents=True)
    (OUT / "tables").mkdir(exist_ok=True, parents=True)
    CKPT = Path(str(OUT) + "_checkpoints"); CKPT.mkdir(parents=True, exist_ok=True)
    P = Path(args.pipeline_dir)
    IU = Path(args.iu_dir)
    print(f"model={args.model_id} out={OUT} pipeline={P} iu={IU}", flush=True)

    splits = pd.read_csv(P / "splits.csv")
    test = splits[splits["split"].eq("test")].copy()
    c2f = pd.read_csv(P / "C2.csv"); c3f = pd.read_csv(P / "C3.csv"); c4f = pd.read_csv(P / "C4.csv")

    proj = pd.read_csv(IU / "indiana_projections.csv")
    img_dir = IU / "images" / "images_normalized"
    fname = proj[proj["projection"].eq("Frontal")].drop_duplicates("uid").set_index("uid")["filename"].astype(str)
    uid_to_path = {u: str(img_dir / f) for u, f in fname.items()}

    def clean(df_: pd.DataFrame) -> pd.DataFrame:
        df_ = df_.copy()
        df_["ret_report"] = df_["ret_report"].fillna("").astype(str)
        df_ = df_[df_["ret_report"].str.len().gt(0)].copy()
        df_["image_path"] = df_["uid"].map(uid_to_path)
        return df_[df_["image_path"].notna()].reset_index(drop=True)

    c2f, c3f, c4f = clean(c2f), clean(c3f), clean(c4f)
    c0f = test[test["uid"].isin(uid_to_path)].copy()
    c0f["image_path"] = c0f["uid"].map(uid_to_path)
    c0f = c0f[["uid", "report_text", "image_path"]].rename(columns={"report_text": "ret_report"}).reset_index(drop=True)
    c0f["ret_report"] = c0f["ret_report"].fillna("").astype(str)
    if args.limit:
        c0f = c0f.head(args.limit); c2f = c2f.head(args.limit)
        c3f = c3f.head(args.limit); c4f = c4f.head(args.limit)
    print(f"C0={len(c0f)} C2={len(c2f)} C3={len(c3f)} C4={len(c4f)}", flush=True)

    # ---------------------------------------------------------- CLIP control
    import open_clip

    HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    cm, _, cp = open_clip.create_model_and_transforms(HUB)
    ct = open_clip.get_tokenizer(HUB)
    cm = cm.to("cuda").eval()

    def ci(path):
        t = cp(Image.open(path).convert("RGB")).unsqueeze(0).to("cuda")
        with torch.no_grad():
            e = cm.encode_image(t)
            return (e / e.norm(dim=-1, keepdim=True)).cpu().float().numpy()[0]

    def ct2(s):
        t = ct([str(s)], context_length=256).to("cuda")
        with torch.no_grad():
            e = cm.encode_text(t)
            return (e / e.norm(dim=-1, keepdim=True)).cpu().float().numpy()[0]

    def n_tokens(s: str) -> int:
        ids = ct([str(s)], context_length=512)[0]
        return int((ids != 0).sum())

    clip_scores = {}
    clip_skipped = {}
    for tag, df_ in [("C0", c0f), ("C2", c2f), ("C3", c3f), ("C4", c4f)]:
        scores, skipped = [], 0
        for _, r in df_.iterrows():
            # Do not let the 256-token BiomedCLIP encoder silently truncate.
            if n_tokens(r["ret_report"]) > 256:
                scores.append(None)
                skipped += 1
                continue
            scores.append(float(np.dot(ci(r["image_path"]), ct2(str(r["ret_report"])))))
        clip_scores[tag] = scores
        clip_skipped[tag] = skipped
        print(f"CLIP {tag}: {len(scores)} scored, {skipped} skipped (>256 tokens)", flush=True)
    del cm
    gc.collect(); torch.cuda.empty_cache()

    # ---------------------------------------------------------- LLaVA
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    proc = AutoProcessor.from_pretrained(args.model_id)
    llava = LlavaForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()
    LAYERS = llava.model.language_model.layers
    N = len(LAYERS)
    IMG_TOK = proc.tokenizer.convert_tokens_to_ids("<image>")
    print(f"loaded {args.model_id}: layers={N} img_tok={IMG_TOK}", flush=True)

    def mp(img_path: str, report: str, mode: str = "baseline"):
        img = Image.open(img_path).convert("RGB")
        prompt = AT.format(finding=report) if mode == "aea" else BT.format(report=report)
        inp = proc(text=prompt, images=img, return_tensors="pt").to("cuda")
        cap = {}

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            cap["h"] = h.detach().float()

        handle = LAYERS[N - 2].register_forward_hook(hook)
        try:
            with torch.no_grad():
                llava(**inp)
        finally:
            handle.remove()
        cls = cap["h"][0, -1].cpu().numpy().astype("float32") if "h" in cap else None
        gen = None
        if mode == "aea":
            with torch.no_grad():
                g = llava.generate(**inp, max_new_tokens=80, do_sample=False)
            gen = proc.decode(g[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return cls, gen

    # ---------------------------------------------------- extraction
    ckpt_path = CKPT / "cls_aea.pkl"
    if ckpt_path.exists():
        d = pickle.load(open(ckpt_path, "rb"))
        cl_f, cl_l, cl_g, ar = d["cl_f"], d["cl_l"], d["cl_g"], d["ar"]
        done = {r["cond"] for r in ar}
        c0_done = len(cl_f) > 0
        print(f"checkpoint loaded: cls={len(cl_f)} aea={len(ar)} done={sorted(done)} c0_done={c0_done}", flush=True)
    else:
        cl_f, cl_l, cl_g, ar = [], [], [], []
        done = set()
        c0_done = False

    if not c0_done:
        for i, r in c0f.iterrows():
            if i % 50 == 0:
                print(f"C0 {i}/{len(c0f)}", flush=True)
            c, _ = mp(r["image_path"], r["ret_report"], mode="baseline")
            if c is not None:
                cl_f.append(c); cl_l.append(0); cl_g.append(str(r["uid"]))
        pickle.dump({"cl_f": cl_f, "cl_l": cl_l, "cl_g": cl_g, "ar": ar}, open(ckpt_path, "wb"))
        print("  checkpoint after C0", flush=True)

    for tag, df_ in [("C2", c2f), ("C3", c3f), ("C4", c4f)]:
        if tag in done:
            print(f"  {tag} already in checkpoint; skipping", flush=True)
            continue
        for i, r in df_.iterrows():
            if i % 50 == 0:
                print(f"{tag} {i}/{len(df_)}", flush=True)
            if tag in ("C2", "C3"):
                cb, _ = mp(r["image_path"], r["ret_report"], mode="baseline")
                if cb is not None:
                    cl_f.append(cb); cl_l.append(1); cl_g.append(str(r["uid"]))
            _, g = mp(r["image_path"], r["ret_report"], mode="aea")
            ar.append({"cond": tag, "uid": r["uid"], "verdict": robust_verdict(g), "gen": g})
        pickle.dump({"cl_f": cl_f, "cl_l": cl_l, "cl_g": cl_g, "ar": ar}, open(ckpt_path, "wb"))
        done.add(tag)
        print(f"  checkpoint after {tag}", flush=True)

    ad = pd.DataFrame(ar, columns=["cond", "uid", "verdict", "gen"])
    ad.to_csv(OUT / "aea_verdicts.csv", index=False)
    X_cls = np.vstack(cl_f).astype("float32")
    y_cls = np.array(cl_l); g_cls = np.array(cl_g)
    np.savez_compressed(OUT / "cls_features.npz", X=X_cls, y=y_cls, groups=g_cls)
    print(f"CLS: {X_cls.shape} pos={int(y_cls.sum())} neg={int((y_cls == 0).sum())}", flush=True)

    # ---------------------------------------------------- CLS probe
    def gna(X, y, g):
        o = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        cfg = [(f"raw_C{c:g}", c, None) for c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]] + [
            (f"pca{n}_C{c:g}", c, n) for n in [16, 32, 64, 128] for c in [1e-3, 1e-2, 1e-1, 1]
        ]

        def mk(C, n_comp):
            steps = [("scale", StandardScaler())]
            if n_comp is not None:
                steps.append(("pca", PCA(n_components=n_comp, random_state=SEED)))
            steps.append(("clf", LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=4000, random_state=SEED)))
            return Pipeline(steps)

        fa = []
        for fold_idx, (tr, te) in enumerate(o.split(X, y, g), 1):
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold_idx)
            isp = list(inner.split(X[tr], y[tr], g[tr]))
            cd = []
            for nm, Cval, n_comp in cfg:
                if n_comp is not None and n_comp > min(X.shape[1], min(len(a) for a, _ in isp)):
                    continue
                try:
                    aa = [
                        roc_auc_score(y[tr][iv], mk(Cval, n_comp).fit(X[tr][it], y[tr][it]).predict_proba(X[tr][iv])[:, 1])
                        for it, iv in isp
                    ]
                    cd.append((nm, Cval, n_comp, float(np.mean(aa))))
                except Exception:
                    continue
            if not cd:
                continue
            b = max(cd, key=lambda r: r[3])
            cl = mk(b[1], b[2]); cl.fit(X[tr], y[tr])
            fa.append(roc_auc_score(y[te], cl.predict_proba(X[te])[:, 1]))
        return float(np.mean(fa)), float(np.std(fa)), fa

    cls_a, cls_s, cls_folds = gna(X_cls, y_cls, g_cls)
    print(f"CLS AUC={cls_a:.4f} +/- {cls_s:.4f} folds={cls_folds}", flush=True)

    c0_clip = [x for x in clip_scores["C0"] if x is not None]
    c2_clip = [x for x in clip_scores["C2"] if x is not None]
    c3_clip = [x for x in clip_scores["C3"] if x is not None]
    y0 = np.concatenate([np.zeros(len(c0_clip)), np.ones(len(c2_clip) + len(c3_clip))])
    cc = np.array(c0_clip + c2_clip + c3_clip)
    auc_clip = roc_auc_score(y0, -cc) if len(set(y0)) > 1 else None
    print(f"CLIP AUC={auc_clip} (skipped={clip_skipped})", flush=True)

    verdicts = ad["verdict"].value_counts().to_dict()
    per_cond = {}
    for cond in ["C2", "C3", "C4"]:
        sub = ad[ad["cond"].eq(cond)]
        n = max(len(sub), 1)
        per_cond[cond] = {
            "n": int(len(sub)),
            **{k: int((sub["verdict"] == k).sum()) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
            **{f"{k}_pct": float((sub["verdict"] == k).mean() * 100) for k in ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]},
        }
    summary = {
        "model": args.model_id,
        "n_layers": N,
        "cls_auc": cls_a,
        "cls_auc_std": cls_s,
        "cls_folds": cls_folds,
        "clip_auc": auc_clip,
        "clip_skipped_over_256_tokens": clip_skipped,
        "verdicts": verdicts,
        "per_condition": per_cond,
        "n": {"C0": len(c0f), "C2": len(c2f), "C3": len(c3f), "C4": len(c4f)},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    # ---------------------------------------------------- figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    conds = ["C2", "C3", "C4"]
    labels = ["SUPPORTS", "CONTRADICTS", "INCONCLUSIVE", "INVALID"]
    colors = {"SUPPORTS": "#2ecc71", "CONTRADICTS": "#e74c3c", "INCONCLUSIVE": "#f39c12", "INVALID": "#95a5a6"}
    bottom = np.zeros(len(conds))
    for lab in labels:
        vals = np.array([per_cond[c][f"{lab}_pct"] for c in conds])
        ax.bar(conds, vals, bottom=bottom, label=lab, color=colors[lab])
        bottom += vals
    ax.set_ylabel("%")
    ax.set_title(f"{args.model_id} verdict distribution")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "fig_verdict_dist.png", dpi=300)
    plt.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
