#!/usr/bin/env python3
"""
Causal Activation Patching — InternVL2-8B (SFT), CLS readout (NO MDI).

Tests whether visual tokens at late layers CAUSALLY drive the internal
matched-vs-mismatched representation (the CLS probe, AUC ~0.80).

Fixes vs old_ref CELL 6:
  - Readout is the CLS probe (logistic regression on layer-30 last-token hidden
    state), NOT MDI.
  - Same image, cross-CONTEXT patch: mismatched = (img, wrong_report),
    matched context = (img, true_report). Only the text differs, so visual-token
    positions are identical across contexts (patch is position-aligned).
  - Both directions + a no-op control (patch a context with itself).
  - Random sampling (not first-50), covers C2/C3/C4 pooled.
  - InternVL2-8B: 256 <IMG_CONTEXT> visual tokens.

Readout: does patching flip P(mismatched) from the CLS probe?
  - matched->mismatched: expect P(mismatched) to DROP (visual tokens causally
    carry the conflict signal; removing it via matched visual states reduces
    the internal mismatch readout).
  - mismatched->matched: expect P(mismatched) to RISE.
  - no-op: expect ~0 change.
"""
import gc, json, random, warnings, pickle
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = 'cuda:0'
OUT = Path('/kaggle/working/causal_patching_paper'); OUT.mkdir(exist_ok=True, parents=True)
CKPT = Path('/kaggle/working/causal_patching_paper_checkpoints'); CKPT.mkdir(exist_ok=True, parents=True)

N_PROBE = 150          # matched + mismatched samples for probe training
N_DIR = 50             # pairs per direction
LAYERS = [20, 25, 28, 30]
READOUT_LAYER = 30     # layer-30 last-token hidden state (matches eval CLS feature)

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
    df_['true_report'] = df_['true_report'].fillna('').astype(str)
    df_ = df_[df_['ret_report'].str.len().gt(0)].copy()
    df_['image_path'] = df_['uid'].map(uid_to_path)
    return df_[df_['image_path'].notna()].reset_index(drop=True)

c2f, c3f, c4f = clean(c2f), clean(c3f), clean(c4f)
mmf = pd.concat([c2f.assign(cond='C2'), c3f.assign(cond='C3'), c4f.assign(cond='C4')], ignore_index=True)
c0f = test[test['uid'].isin(uid_to_path)].copy()
c0f['image_path'] = c0f['uid'].map(uid_to_path)
c0f = c0f[['uid','report_text','image_path']].rename(columns={'report_text':'true_report'}).reset_index(drop=True)
c0f['true_report'] = c0f['true_report'].fillna('').astype(str)
print(f'matched(C0)={len(c0f)} mismatched(C2/C3/C4)={len(mmf)}', flush=True)

# --- Load InternVL2-8B (SFT) on GPU 0 ---
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
try:
    IL = ivl.language_model.model.layers
except AttributeError:
    IL = ivl.language_model.layers
N = len(IL)
print(f'InternVL2-8B loaded on {DEV}. {N} layers.', flush=True)

IP_BASE = ('<|im_start|>user\n<image>\nRetrieved record: {report}\nDiagnosis based on image and record?<|im_end|>\n<|im_start|>assistant\n')
tf = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def build_inp(img_path, report):
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
    return inp

def forward(img_path, report, patch_layer=None, cached_vis=None, cache_layers=None):
    """One forward pass; returns CLS feature (layer-30 last token) + optional visual cache."""
    inp = build_inp(img_path, report)
    if inp is None:
        return None, None
    ids = inp['input_ids'][0]
    vis_pos = (ids == ivl.img_context_token_id).nonzero(as_tuple=True)[0]
    if len(vis_pos) == 0:
        return None, None
    vis_pos = vis_pos.to(DEV)

    cache = {}
    hooks = []
    if cache_layers is not None:
        def make_cache(l):
            def fn(mod, hi, out):
                o = out[0] if isinstance(out, tuple) else out
                cache[l] = o[0, vis_pos].detach().clone()
            return fn
        for l in cache_layers:
            hooks.append(IL[l].register_forward_hook(make_cache(l)))
    if patch_layer is not None:
        cv = cached_vis
        def make_patch(l):
            def fn(mod, hi, out):
                o = out[0] if isinstance(out, tuple) else out
                new = o.clone()
                new[0, vis_pos] = cv.to(new.device, new.dtype)
                if isinstance(out, tuple):
                    return (new,) + out[1:]
                return new
            return fn
        hooks.append(IL[patch_layer].register_forward_hook(make_patch(patch_layer)))
    try:
        with torch.no_grad():
            out = ivl(**inp, output_hidden_states=True, return_dict=True)
        cls = out.hidden_states[-2][0, -1].float().cpu().numpy().astype('float32')
    finally:
        for h in hooks:
            h.remove()
    return cls, cache

# --- Step 0: train CLS probe ---
probe_ckpt = CKPT / 'probe.pkl'
if probe_ckpt.exists():
    sc, clf, probe_auc = pickle.load(open(probe_ckpt, 'rb'))
    print(f'Loaded probe. AUC={probe_auc:.4f}', flush=True)
else:
    m_probe = c0f.sample(min(N_PROBE, len(c0f)), random_state=SEED).reset_index(drop=True)
    mm_probe = mmf.sample(min(N_PROBE, len(mmf)), random_state=SEED).reset_index(drop=True)
    X, y = [], []
    for i, (_, r) in enumerate(m_probe.iterrows()):
        if i % 50 == 0: print(f'probe matched {i}/{len(m_probe)}', flush=True)
        cls, _ = forward(r['image_path'], r['true_report'])
        if cls is not None:
            X.append(cls); y.append(0)
    for i, (_, r) in enumerate(mm_probe.iterrows()):
        if i % 50 == 0: print(f'probe mismatched {i}/{len(mm_probe)}', flush=True)
        cls, _ = forward(r['image_path'], r['ret_report'])
        if cls is not None:
            X.append(cls); y.append(1)
    X = np.vstack(X).astype('float32'); y = np.array(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=4000, random_state=SEED).fit(sc.transform(X), y)
    probe_auc = roc_auc_score(y, clf.predict_proba(sc.transform(X))[:, 1])
    pickle.dump((sc, clf, probe_auc), open(probe_ckpt, 'wb'))
    print(f'Probe trained: n={len(y)} AUC={probe_auc:.4f}', flush=True)

def score(cls):
    return float(clf.predict_proba(sc.transform(cls.reshape(1, -1)))[:, 1][0])

# --- Step 1: patching ---
rows = []
results = {}

def run_direction(name, source_df, source_report_col, target_df, target_report_col):
    src = source_df.sample(min(N_DIR, len(source_df)), random_state=SEED).reset_index(drop=True)
    tgt = target_df.sample(min(N_DIR, len(target_df)), random_state=SEED).reset_index(drop=True)
    n = min(len(src), len(tgt))
    print(f'\n=== {name}: {n} pairs ===', flush=True)
    per_layer = {l: {'baseline': [], 'patched': []} for l in LAYERS}
    for i in range(n):
        if i % 10 == 0: print(f'  {name} {i}/{n}', flush=True)
        sr, tr = src.iloc[i], tgt.iloc[i]
        # target baseline (unpatched)
        b_cls, _ = forward(tr['image_path'], tr[target_report_col])
        if b_cls is None:
            continue
        b_score = score(b_cls)
        # source cache (visual at all layers, single forward)
        _, cache = forward(sr['image_path'], sr[source_report_col], cache_layers=LAYERS)
        if not cache:
            continue
        for l in LAYERS:
            p_cls, _ = forward(tr['image_path'], tr[target_report_col], patch_layer=l, cached_vis=cache[l])
            if p_cls is None:
                continue
            p_score = score(p_cls)
            per_layer[l]['baseline'].append(b_score)
            per_layer[l]['patched'].append(p_score)
            rows.append({'direction': name, 'layer': l, 'uid': tr['uid'], 'baseline_score': b_score, 'patched_score': p_score})
        if i % 20 == 0 and i > 0:
            pd.DataFrame(rows).to_csv(OUT / 'patching_rows.csv', index=False)
    out = {}
    for l in LAYERS:
        b = np.array(per_layer[l]['baseline']); p = np.array(per_layer[l]['patched'])
        if len(b) == 0:
            continue
        out[l] = {'baseline_mean': float(b.mean()), 'patched_mean': float(p.mean()),
                  'delta': float((p - b).mean()), 'n': int(len(b))}
        print(f'  L{l}: baseline P(mismatch)={b.mean():.3f} -> patched={p.mean():.3f}  delta={out[l]["delta"]:+.3f} (n={len(b)})', flush=True)
    results[name] = out

# direction 1: matched -> mismatched
run_direction('matched_to_mismatched', c0f, 'true_report', mmf, 'ret_report')
# direction 2: mismatched -> matched
run_direction('mismatched_to_matched', mmf, 'ret_report', c0f, 'true_report')

# no-op control: mismatched -> mismatched (patch with itself)
def run_noop():
    tgt = mmf.sample(min(N_DIR, len(mmf)), random_state=SEED).reset_index(drop=True)
    print(f'\n=== noop control: {len(tgt)} pairs ===', flush=True)
    per_layer = {l: {'baseline': [], 'patched': []} for l in LAYERS}
    for i, (_, tr) in enumerate(tgt.iterrows()):
        if i % 10 == 0: print(f'  noop {i}/{len(tgt)}', flush=True)
        b_cls, cache = forward(tr['image_path'], tr['ret_report'], cache_layers=LAYERS)
        if b_cls is None or not cache:
            continue
        b_score = score(b_cls)
        for l in LAYERS:
            p_cls, _ = forward(tr['image_path'], tr['ret_report'], patch_layer=l, cached_vis=cache[l])
            if p_cls is None:
                continue
            per_layer[l]['baseline'].append(b_score)
            per_layer[l]['patched'].append(score(p_cls))
    out = {}
    for l in LAYERS:
        b = np.array(per_layer[l]['baseline']); p = np.array(per_layer[l]['patched'])
        if len(b) == 0:
            continue
        out[l] = {'baseline_mean': float(b.mean()), 'patched_mean': float(p.mean()),
                  'delta': float((p - b).mean()), 'n': int(len(b))}
        print(f'  L{l}: baseline={b.mean():.3f} -> patched={p.mean():.3f}  delta={out[l]["delta"]:+.4f} (n={len(b)})', flush=True)
    results['noop'] = out

run_noop()

pd.DataFrame(rows).to_csv(OUT / 'patching_rows.csv', index=False)
final = {'model': 'InternVL2-8B', 'probe_auc': probe_auc, 'layers': LAYERS, 'readout_layer': READOUT_LAYER, 'by_direction': results}
json.dump(final, open(OUT / 'causal_patching_results.json', 'w'), indent=2)
print('\n' + json.dumps(final, indent=2), flush=True)
print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
