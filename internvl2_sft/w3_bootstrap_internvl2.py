#!/usr/bin/env python3
"""
W3 — Bootstrap CI + permutation test for the CLS matched-vs-mismatched signal
(InternVL2-8B SFT), no MDI.

Addresses reviewer W3 ("C0 vs mismatch p=0.206 -> strengthen statistical power"):
  - Larger matched (C0) + mismatched (C2/C3/C4) CLS feature sample.
  - Bootstrapped 95% CI on the AUROC.
  - Permutation test (label shuffle) for significance.

CLS feature = layer-30 last-token hidden state (baseline prompt), as in the
behavioral eval (AUC 0.798).
"""
import gc, json, random, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = 'cuda:0'
OUT = Path('/kaggle/working/w3_bootstrap_paper'); OUT.mkdir(exist_ok=True, parents=True)

N_MATCHED = 400
N_MISMATCHED = 400
N_BOOT = 500
N_PERM = 500

# --- Load corrected pipeline data ---
PREV = next((p for p in [
    Path('/kaggle/working/aea-xai-pipeline'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/corrected-aea-xai-pipeline-output'),
    Path('/kaggle/input/corrected-aea-xai-pipeline-output'),
] if p.is_dir()), None)
if PREV is None:
    raise FileNotFoundError('corrected pipeline output not found')
splits = pd.read_csv(PREV / 'splits.csv')
c2f = pd.read_csv(PREV / 'C2.csv'); c3f = pd.read_csv(PREV / 'C3.csv'); c4f = pd.read_csv(PREV / 'C4.csv')
test = splits[splits['split'].eq('test')].copy()

IU = next((p for p in [
    Path('/kaggle/input/datasets/raddar/chest-xrays-indiana-university'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/chest-xrays-indiana-university'),
    Path('/kaggle/input/chest-xrays-indiana-university'),
] if p.is_dir()), None)
proj = pd.read_csv(IU / 'indiana_projections.csv')
img_dir = next((p for p in [IU/'images'/'images_normalized', IU/'images', IU] if p.is_dir()), IU)
all_img_files = list(img_dir.rglob('*'))
fname_to_path = {p.name: p for p in all_img_files if p.is_file()}
uid_to_path = {}
for _, r in proj[proj['projection'].eq('Frontal')].iterrows():
    fn = str(r['filename'])
    if fn in fname_to_path:
        uid_to_path[r['uid']] = fname_to_path[fn]

def clean(df_):
    df_ = df_.copy()
    df_['ret_report'] = df_['ret_report'].fillna('').astype(str)
    df_ = df_[df_['ret_report'].str.len().gt(0)].copy()
    df_['image_path'] = df_['uid'].map(uid_to_path)
    return df_[df_['image_path'].notna()].reset_index(drop=True)

c2f, c3f, c4f = clean(c2f), clean(c3f), clean(c4f)
mmf = pd.concat([c2f.assign(cond='C2'), c3f.assign(cond='C3'), c4f.assign(cond='C4')], ignore_index=True)
c0f = test[test['uid'].isin(uid_to_path)].copy()
c0f['image_path'] = c0f['uid'].map(uid_to_path)
c0f = c0f[['uid','report_text','image_path']].rename(columns={'report_text':'report'}).reset_index(drop=True)
c0f['report'] = c0f['report'].fillna('').astype(str)
mmf = mmf.rename(columns={'ret_report': 'report'})
print(f'matched(C0)={len(c0f)} mismatched={len(mmf)}', flush=True)

# --- Load InternVL2-8B (SFT) ---
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
LOCAL = '/root/models/internvl2_sft_model'
import pathlib
if not pathlib.Path(LOCAL).exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='OpenGVLab/InternVL2-8B', local_dir=LOCAL)
_vit = pathlib.Path(LOCAL) / 'modeling_intern_vit.py'
if _vit.exists():
    _t = _vit.read_text()
    if 'dpr = [x.item()' in _t:
        _t = _t.replace(
            "dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]",
            "dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) for i in range(config.num_hidden_layers)]"
        )
        _vit.write_text(_t)
itok = AutoTokenizer.from_pretrained(LOCAL, trust_remote_code=True)
ivl = AutoModel.from_pretrained(LOCAL, torch_dtype=torch.float16, load_in_8bit=True,
                                trust_remote_code=True, device_map={'': DEV}).eval()
ivl.img_context_token_id = itok.convert_tokens_to_ids('<IMG_CONTEXT>')
print(f'InternVL2-8B loaded on {DEV}.', flush=True)

IP_BASE = ('<|im_start|>user\n<image>\nRetrieved record: {report}\nDiagnosis based on image and record?<|im_end|>\n<|im_start|>assistant\n')
tf = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def cls_feature(img_path, report):
    if not isinstance(report, str) or not report.strip():
        return None
    pr = IP_BASE.format(report=str(report))
    img_tok = '<img>' + '<IMG_CONTEXT>' * 256 + '</img>'
    pr = pr.replace('<image>', img_tok)
    inp = itok(pr, return_tensors='pt').to(DEV)
    try:
        inp['pixel_values'] = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEV).half()
    except Exception:
        return None
    inp['image_flags'] = torch.tensor([1], dtype=torch.long, device=DEV)
    try:
        with torch.no_grad():
            out = ivl(**inp, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-2][0, -1].float().cpu().numpy().astype('float32')
    except Exception:
        return None

# --- Extract features ---
matched = c0f.sample(min(N_MATCHED, len(c0f)), random_state=SEED).reset_index(drop=True)
mismatched = mmf.sample(min(N_MISMATCHED, len(mmf)), random_state=SEED).reset_index(drop=True)
X, y = [], []
for i, (_, r) in enumerate(matched.iterrows()):
    if i % 100 == 0: print(f'matched {i}/{len(matched)}', flush=True)
    c = cls_feature(r['image_path'], r['report'])
    if c is not None:
        X.append(c); y.append(0)
for i, (_, r) in enumerate(mismatched.iterrows()):
    if i % 100 == 0: print(f'mismatched {i}/{len(mismatched)}', flush=True)
    c = cls_feature(r['image_path'], r['report'])
    if c is not None:
        X.append(c); y.append(1)
X = np.vstack(X).astype('float32'); y = np.array(y)
np.savez_compressed(OUT / 'w3_cls_features.npz', X=X, y=y)
print(f'Features: X={X.shape} n_matched={(y==0).sum()} n_mismatched={(y==1).sum()}', flush=True)

# --- Pipeline: StandardScaler + LogisticRegression ---
def fit_score(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=4000, random_state=SEED)
    clf.fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])

# observed 5-fold CV AUC
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_aucs = [fit_score(X[tr], y[tr], X[te], y[te]) for tr, te in skf.split(X, y)]
obs_auc = float(np.mean(cv_aucs))
obs_std = float(np.std(cv_aucs))
print(f'Observed 5-fold CV AUROC = {obs_auc:.4f} +/- {obs_std:.4f}', flush=True)

# --- Bootstrap CI (resample samples with replacement) ---
n = len(y)
rng = np.random.default_rng(SEED)
boot_aucs = []
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    Xb, yb = X[idx], y[idx]
    if len(np.unique(yb)) < 2:
        continue
    try:
        boot_aucs.append(fit_score(Xb, yb, Xb, yb))
    except Exception:
        continue
boot_aucs = np.array(boot_aucs)
lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
print(f'Bootstrap 95% CI = [{lo:.4f}, {hi:.4f}] (n={len(boot_aucs)})', flush=True)

# --- Permutation test (shuffle labels, in-sample AUROC) ---
perm_aucs = []
for p_ in range(N_PERM):
    yp = rng.permutation(y)
    try:
        perm_aucs.append(fit_score(X, yp, X, yp))
    except Exception:
        continue
perm_aucs = np.array(perm_aucs)
pval = float((perm_aucs >= obs_auc).mean())
print(f'Permutation test: observed={obs_auc:.4f}, null mean={perm_aucs.mean():.4f}, p={pval:.4f}', flush=True)

results = {
    'model': 'InternVL2-8B',
    'n_matched': int((y == 0).sum()),
    'n_mismatched': int((y == 1).sum()),
    'observed_cv_auc': obs_auc,
    'observed_cv_std': obs_std,
    'cv_folds': cv_aucs,
    'bootstrap_auc_mean': float(boot_aucs.mean()),
    'bootstrap_ci_95': [float(lo), float(hi)],
    'n_bootstrap': int(len(boot_aucs)),
    'permutation_null_mean': float(perm_aucs.mean()),
    'permutation_pvalue': pval,
    'n_permutations': int(len(perm_aucs)),
}
json.dump(results, open(OUT / 'w3_results.json', 'w'), indent=2)
print('\n' + json.dumps(results, indent=2), flush=True)
print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
