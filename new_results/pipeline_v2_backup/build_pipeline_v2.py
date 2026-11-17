#!/usr/bin/env python3
"""
Rebuild the AEA-XAI condition pipeline with:
  - frontal projections only
  - exact MeSH/Problems labels (normal iff both contain the exact token "normal")
  - no text truncation anywhere

Outputs (in --out-dir):
  splits.csv, retrieval.csv, rivn.csv, cmcg.json,
  C1.csv, C2.csv, C3.csv, C4.csv,
  pipeline_audit.json, fig_roc.png

This script only builds conditions and retrieval/calibration metadata. It does
not load a VLM.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

SEED = 42
HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_normal_field(value) -> bool:
    """Exact token match, not substring. Avoids 'abnormal' containing 'normal'."""
    return any(part.strip().lower() == "normal" for part in str(value).split(";"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--iu-dir",
        default="/kaggle/input/datasets/raddar/chest-xrays-indiana-university",
    )
    ap.add_argument("--out-dir", default="/kaggle/working/aea-xai-pipeline-v2")
    args = ap.parse_args()

    set_seed(SEED)
    iu = Path(args.iu_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} out={out}", flush=True)

    # ---------------------------------------------------------------- data
    reports = pd.read_csv(iu / "indiana_reports.csv")
    projections = pd.read_csv(iu / "indiana_projections.csv")

    frontal = projections[projections["projection"].eq("Frontal")].copy()
    df = (
        frontal.merge(
            reports[["uid", "MeSH", "Problems", "findings", "impression"]],
            on="uid",
            how="inner",
        )
        .drop_duplicates("uid")
        .reset_index(drop=True)
    )
    for col in ("findings", "impression"):
        df[col] = df[col].fillna("").astype(str)
    df["report_text"] = (df["findings"] + " " + df["impression"]).str.strip()

    # Exact label rule. "No Indexing" is not an exact match for "normal", so it
    # is treated as abnormal (matching the 738-row test set / C0 count).
    df["label"] = np.where(
        df["MeSH"].map(is_normal_field) & df["Problems"].map(is_normal_field),
        "normal",
        "abnormal",
    )

    # Image paths from the frontal projection table (not globbing).
    fname = (
        frontal.drop_duplicates("uid").set_index("uid")["filename"].astype(str)
    )
    df["image_filename"] = df["uid"].map(fname)
    df["filename"] = df["image_filename"]  # legacy column name
    df["projection"] = "Frontal"
    img_dir = iu / "images" / "images_normalized"
    df["image_path"] = df["image_filename"].map(lambda f: str(img_dir / f))
    df["image_exists"] = df["image_path"].map(lambda p: Path(p).is_file())

    # Same split as the original pipeline.
    tr, te = train_test_split(df["uid"].values, test_size=0.20, random_state=SEED)
    tr, va = train_test_split(tr, test_size=0.125, random_state=SEED)
    df["split"] = "train"
    df.loc[df["uid"].isin(va), "split"] = "val"
    df.loc[df["uid"].isin(te), "split"] = "test"

    train_df = df[df["split"].eq("train")].reset_index(drop=True)
    val_df = df[df["split"].eq("val")].reset_index(drop=True)
    test_df = df[df["split"].eq("test")].reset_index(drop=True)

    df.to_csv(out / "splits.csv", index=False)
    print(
        f"frontal reports={len(df)} (normal={int(df.label.eq('normal').sum())}, "
        f"abnormal={int(df.label.eq('abnormal').sum())})",
        flush=True,
    )
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(
            f"{name}: n={len(d)} normal={int(d.label.eq('normal').sum())} "
            f"abnormal={int(d.label.eq('abnormal').sum())} "
            f"images_missing={int((~d.image_exists).sum())}",
            flush=True,
        )

    # ------------------------------------------------------- BiomedCLIP
    import open_clip

    clip, _, prep = open_clip.create_model_and_transforms(HUB)
    tokenizer = open_clip.get_tokenizer(HUB)
    clip = clip.to(dev).eval()

    def encode_texts(texts: list[str], bs: int = 64) -> np.ndarray:
        embs: list[np.ndarray] = []
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            toks = tokenizer(batch, context_length=256).to(dev)
            with torch.no_grad():
                e = clip.encode_text(toks)
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().float().numpy())
        return np.vstack(embs)

    def encode_images(uids: list[str], bs: int = 32) -> tuple[np.ndarray, list[str]]:
        by_uid = df.set_index("uid")
        embs: list[np.ndarray] = []
        valid: list[str] = []
        for i in range(0, len(uids), bs):
            batch, kept = [], []
            for u in uids[i : i + bs]:
                p = by_uid.loc[u, "image_path"]
                if not Path(p).is_file():
                    continue
                try:
                    batch.append(prep(Image.open(p).convert("RGB")))
                    kept.append(u)
                except Exception as exc:  # noqa: BLE001
                    print(f"image read failed uid={u}: {exc}", flush=True)
            if not batch:
                continue
            with torch.no_grad():
                e = clip.encode_image(torch.stack(batch).to(dev))
                e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu().float().numpy())
            valid.extend(kept)
        return np.vstack(embs), valid

    # Token audit: assert no train report exceeds the BiomedCLIP text limit.
    def max_tokens(texts: list[str], bs: int = 256) -> int:
        m = 0
        for i in range(0, len(texts), bs):
            ids = tokenizer(texts[i : i + bs], context_length=512)
            m = max(m, max(int((row != 0).sum()) for row in ids))
        return m

    train_texts = train_df["report_text"].tolist()
    max_train_tokens = max_tokens(train_texts)
    max_all_tokens = max_tokens(df["report_text"].tolist())
    assert max_train_tokens <= 256, (
        f"BiomedCLIP would truncate a train report: max_train_tokens={max_train_tokens}"
    )
    print(
        f"BiomedCLIP tokens: train_max={max_train_tokens} all_max={max_all_tokens} "
        f"(train corpus is not truncated)",
        flush=True,
    )

    print("encoding train reports...", flush=True)
    corp_emb = encode_texts(train_texts)
    np.save(out / "corp_emb.npy", corp_emb)

    print("encoding test images...", flush=True)
    test_emb, test_uids = encode_images(test_df["uid"].tolist())
    np.save(out / "test_emb.npy", test_emb)
    if len(test_uids) != len(test_df):
        print(
            f"WARNING: only {len(test_uids)}/{len(test_df)} test images encoded",
            flush=True,
        )

    index = faiss.IndexFlatIP(512)
    index.add(corp_emb)
    faiss.write_index(index, str(out / "faiss.index"))
    D, I = index.search(test_emb, k=3)

    ret_rows = []
    for i, u in enumerate(test_uids):
        for rank in range(3):
            j = int(I[i, rank])
            ret_rows.append(
                {
                    "query": u,
                    "rank": rank + 1,
                    "ret_uid": train_df.iloc[j]["uid"],
                    "sim": float(D[i, rank]),
                    "ret_report": train_df.iloc[j]["report_text"],
                    "ret_label": train_df.iloc[j]["label"],
                }
            )
    ret_df = pd.DataFrame(ret_rows)
    ret_df.to_csv(out / "retrieval.csv", index=False)
    print(f"retrieval rows={len(ret_df)}", flush=True)

    # ------------------------------------------------- RIVN calibration
    print("encoding val images...", flush=True)
    val_emb, val_uids = encode_images(val_df["uid"].tolist())
    val_D, val_I = index.search(val_emb, k=1)
    label_by_uid = df.set_index("uid")["label"].to_dict()
    labels = [
        int(train_df.iloc[int(val_I[i, 0])]["label"] != label_by_uid[u])
        for i, u in enumerate(val_uids)
    ]
    sims = [float(val_D[i, 0]) for i in range(len(val_uids))]
    if len(set(labels)) < 2 and len(labels) > 0:
        labels[-1] = 1 - labels[0]

    best_tau, best_f1 = 0.30, 0.0
    for tau in np.arange(0.10, 0.70, 0.01):
        preds = [1 if s < tau else 0 for s in sims]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    auc = roc_auc_score(labels, [-s for s in sims])
    json.dump(
        {"tau": float(best_tau), "f1": float(best_f1), "auc": float(auc)},
        open(out / "cmcg.json", "w"),
    )
    print(f"CMCG tau={best_tau:.2f} f1={best_f1:.3f} auc={auc:.3f}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, [-s for s in sims])
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("CMCG ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_roc.png", dpi=150)
    plt.close()

    # ---------------------------------------------------- conditions
    test_vdf = test_df[test_df["uid"].isin(test_uids)].reset_index(drop=True)
    max_sims = {u: float(D[i, 0]) for i, u in enumerate(test_uids)}
    rivn = {u: int(max_sims[u] < best_tau) for u in test_uids}
    pd.DataFrame(
        [{"uid": u, "sim": max_sims[u], "rivn": rivn[u]} for u in test_uids]
    ).to_csv(out / "rivn.csv", index=False)

    normal_uids = test_vdf[test_vdf["label"].eq("normal")]["uid"].tolist()
    abnormal_uids = test_vdf[test_vdf["label"].eq("abnormal")]["uid"].tolist()
    abn_train = train_df[
        train_df["label"].eq("abnormal") & train_df["report_text"].str.len().gt(0)
    ].reset_index(drop=True)
    nrm_train = train_df[
        train_df["label"].eq("normal") & train_df["report_text"].str.len().gt(0)
    ].reset_index(drop=True)
    print(
        f"test normal={len(normal_uids)} abnormal={len(abnormal_uids)}; "
        f"train pools: abnormal={len(abn_train)} normal={len(nrm_train)}",
        flush=True,
    )

    def make_pairs(uids, cond, wrong_pool=None):
        rows = []
        if wrong_pool is not None:
            pool = wrong_pool.sample(frac=1, random_state=SEED).reset_index(drop=True)
            for j, u in enumerate(uids):
                row = test_vdf[test_vdf["uid"].eq(u)].iloc[0]
                wrong = pool.iloc[j % len(pool)]
                rows.append(
                    {
                        "uid": u,
                        "cond": cond,
                        "ret_report": wrong["report_text"],
                        "ret_uid": wrong["uid"],
                        "true_report": row["report_text"],
                        "image_label": row["label"],
                        "ret_label": wrong["label"],
                    }
                )
        else:
            for u in uids:
                row = test_vdf[test_vdf["uid"].eq(u)].iloc[0]
                rows.append(
                    {
                        "uid": u,
                        "cond": cond,
                        "ret_report": row["report_text"],
                        "ret_uid": u,
                        "true_report": row["report_text"],
                        "image_label": row["label"],
                        "ret_label": row["label"],
                    }
                )
        return pd.DataFrame(rows)

    c1 = make_pairs(test_uids, "C1")
    c2 = make_pairs(normal_uids, "C2", abn_train)
    c3 = make_pairs(abnormal_uids, "C3", nrm_train)

    c4_rows = []
    abnormal_set = set(abnormal_uids)
    for i, u in enumerate(test_uids):
        if u not in abnormal_set:
            continue
        for rank in range(3):
            rrow = train_df.iloc[int(I[i, rank])]
            if rrow["label"] == "normal" and str(rrow["report_text"]).strip():
                row = test_vdf[test_vdf["uid"].eq(u)].iloc[0]
                c4_rows.append(
                    {
                        "uid": u,
                        "cond": "C4",
                        "ret_report": rrow["report_text"],
                        "ret_uid": rrow["uid"],
                        "true_report": row["report_text"],
                        "image_label": row["label"],
                        "ret_label": rrow["label"],
                    }
                )
                break
    c4 = pd.DataFrame(c4_rows)

    for cond, frame in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4)):
        frame.to_csv(out / f"{cond}.csv", index=False)
        print(
            f"{cond}: n={len(frame)} unique_ret_reports={frame['ret_report'].nunique()} "
            f"unique_ret_uids={frame['ret_uid'].nunique()}",
            flush=True,
        )

    # --------------------------------------------------------- audit
    def purity(frame):
        return {
            "n": int(len(frame)),
            "image_normal_ret_abnormal": int(
                ((frame.image_label == "normal") & (frame.ret_label == "abnormal")).sum()
            ),
            "image_abnormal_ret_normal": int(
                ((frame.image_label == "abnormal") & (frame.ret_label == "normal")).sum()
            ),
            "image_normal_ret_normal": int(
                ((frame.image_label == "normal") & (frame.ret_label == "normal")).sum()
            ),
            "image_abnormal_ret_abnormal": int(
                ((frame.image_label == "abnormal") & (frame.ret_label == "abnormal")).sum()
            ),
        }

    audit = {
        "seed": SEED,
        "iu_dir": str(iu),
        "out_dir": str(out),
        "frontal_reports": int(len(df)),
        "label_counts_all": {
            "normal": int(df.label.eq("normal").sum()),
            "abnormal": int(df.label.eq("abnormal").sum()),
        },
        "label_counts_by_split": {
            name: {
                "n": int(len(d)),
                "normal": int(d.label.eq("normal").sum()),
                "abnormal": int(d.label.eq("abnormal").sum()),
            }
            for name, d in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "images_missing_by_split": {
            name: int((~d.image_exists).sum())
            for name, d in (("train", train_df), ("val", val_df), ("test", test_df))
        },
        "biomedclip_tokens": {
            "train_max": int(max_train_tokens),
            "all_max": int(max_all_tokens),
            "train_truncated": bool(max_train_tokens > 256),
        },
        "train_pools": {
            "abnormal": int(len(abn_train)),
            "normal": int(len(nrm_train)),
        },
        "conditions": {
            name: purity(frame)
            for name, frame in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4))
        },
        "rivn": {
            "tau": float(best_tau),
            "f1": float(best_f1),
            "auc": float(auc),
            "flagged": int(sum(rivn.values())),
            "n": int(len(rivn)),
        },
        "no_indexing_rows": int(
            (
                reports["MeSH"].astype(str).str.contains("No Indexing", case=False, regex=False)
                | reports["Problems"].astype(str).str.contains("No Indexing", case=False, regex=False)
            ).sum()
        ),
    }
    json.dump(audit, open(out / "pipeline_audit.json", "w"), indent=2)
    print("audit:", json.dumps(audit["conditions"], indent=2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
