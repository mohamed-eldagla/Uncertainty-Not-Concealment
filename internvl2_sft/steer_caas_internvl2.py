#!/usr/bin/env python3
"""
CAAS: Conflict-Aware Activation Steering — InternVL2-8B (SFT), v3 (resilient)

Combines direction extraction + steering sweep + false-positive control.
Every generation's RAW TEXT is saved; verdicts computed BOTH naively and robustly
(robust = first-verdict parse, matches offline behavioral reclassification).

Resilience:
  - direction.pkl cached on disk (downloadable).
  - sweep rows flushed to steering_generations_raw.csv every 10 samples, and
    resumed on restart (skips (uid, tag, alpha) already present).
"""
import gc, json, random, re, warnings, pickle
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path('/kaggle/working/caas_internvl2_paper'); OUT.mkdir(exist_ok=True, parents=True)
(OUT / 'figures').mkdir(exist_ok=True, parents=True)
CKPT = Path('/kaggle/working/caas_internvl2_paper_checkpoints'); CKPT.mkdir(exist_ok=True, parents=True)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)

# --- Config ---
N_DIR = 200
N_STEER = 60
N_FP = 30
ALPHAS = [3.0, 5.0, 8.0, 12.0, 15.0, 20.0]
FP_ALPHA = 20.0

# --- verdict parsers ---
_RE_VID = re.compile(r'VISUAL EVIDENCE\s*:?\s*\[?\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)')
_RE_BARE = re.compile(r'^\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\b')

def naive_verdict(gen):
    t = (gen or '').upper()
    return 'CONTRADICTS' if 'CONTRADICT' in t else 'SUPPORTS' if 'SUPPORT' in t else 'INCONCLUSIVE'

def robust_verdict(gen):
    if not gen:
        return 'INVALID'
    t = gen.strip().upper()
    first_line = t.splitlines()[0] if t else ''
    if all(x in first_line for x in ('SUPPORTS', 'CONTRADICTS', 'INCONCLUSIVE')) and ('/' in first_line or '|' in first_line):
        return 'INVALID'
    m = _RE_VID.search(t)
    if m:
        return m.group(1)
    m = _RE_BARE.match(t)
    if m:
        return m.group(1)
    return 'INVALID'

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
c0f = test[test['uid'].isin(uid_to_path)].copy()
c0f['image_path'] = c0f['uid'].map(uid_to_path)
c0f = c0f[['uid','report_text','image_path']].rename(columns={'report_text':'ret_report'}).reset_index(drop=True)
c0f['ret_report'] = c0f['ret_report'].fillna('').astype(str)
mmf = pd.concat([c2f.assign(cond='C2'), c3f.assign(cond='C3'), c4f.assign(cond='C4')], ignore_index=True)
print(f'C0={len(c0f)} mismatched={len(mmf)}', flush=True)

# --- Load InternVL2-8B (SFT) ---
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
LOCAL = '/root/models/internvl2_sft_model'
import pathlib
if not pathlib.Path(LOCAL).exists():
    print(f'Downloading InternVL2-8B to {LOCAL} ...', flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='OpenGVLab/InternVL2-8B', local_dir=LOCAL)
    print('Download done.', flush=True)
_vit = pathlib.Path(LOCAL) / 'modeling_intern_vit.py'
if _vit.exists():
    _t = _vit.read_text()
    if 'dpr = [x.item()' in _t:
        _t = _t.replace(
            "dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]",
            "dpr = [config.drop_path_rate * i / max(config.num_hidden_layers - 1, 1) for i in range(config.num_hidden_layers)]"
        )
        _vit.write_text(_t)
        print('Patched modeling_intern_vit.py dpr .item() bug.', flush=True)
itok = AutoTokenizer.from_pretrained(LOCAL, trust_remote_code=True)
ivl = AutoModel.from_pretrained(LOCAL, torch_dtype=torch.float16, load_in_8bit=True, trust_remote_code=True).eval()
ivl.img_context_token_id = itok.convert_tokens_to_ids('<IMG_CONTEXT>')
try:
    IL = ivl.language_model.model.layers
except AttributeError:
    IL = ivl.language_model.layers
N = len(IL)
print(f'InternVL2-8B loaded. {N} layers.', flush=True)

CRITICAL_LAYERS = list(range(25, N))
IP_AEA = ('<|im_start|>user\n<image>\nRetrieved record suggests: {finding}.\n'
    'Does this X-ray SUPPORT, CONTRADICT, or is INCONCLUSIVE about \'{finding}\'?\n'
    'Check: (a) lung opacification (b) costophrenic angles (c) cardiac silhouette (d) pleural space.\n'
    'Reply ONLY:\nVISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]\nRATIONALE: [one sentence]<|im_end|>\n<|im_start|>assistant\n')
IP_BASE = ('<|im_start|>user\n<image>\nRetrieved record: {report}\nDiagnosis based on image and record?<|im_end|>\n<|im_start|>assistant\n')
tf = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def build_inp(img_path, report, mode='baseline'):
    if not isinstance(report, str) or not report.strip():
        return None
    if mode == 'aea':
        pr = IP_AEA.format(finding=report)
    else:
        pr = IP_BASE.format(report=str(report))
    img_tok = '<img>' + '<IMG_CONTEXT>' * 256 + '</img>'
    pr = pr.replace('<image>', img_tok)
    inp = itok(pr, return_tensors='pt').to('cuda')
    try:
        inp['pixel_values'] = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to('cuda').half()
    except Exception:
        return None
    inp['image_flags'] = torch.tensor([1], dtype=torch.long, device='cuda')
    return inp

def extract_hidden(img_path, report):
    inp = build_inp(img_path, report, mode='baseline')
    if inp is None:
        return {}
    store = {}
    def hk(l):
        def fn(mod, hi, out):
            o = out[0] if isinstance(out, tuple) else out
            store[l] = o[0, -1].detach().float().cpu()
        return fn
    hooks = [IL[l].register_forward_hook(hk(l)) for l in CRITICAL_LAYERS]
    try:
        with torch.no_grad():
            ivl(**inp)
    finally:
        for h in hooks:
            h.remove()
    return store

def generate_steered(img_path, report, directions=None, alpha=0.0):
    inp = build_inp(img_path, report, mode='aea')
    if inp is None:
        return None
    hooks = []
    if directions is not None:
        def shk(l):
            def fn(mod, hi, out):
                o = out[0] if isinstance(out, tuple) else out
                d = directions[l].to(o.device).to(o.dtype)
                o.add_(alpha * d.unsqueeze(0).unsqueeze(0))
                return None
            return fn
        hooks = [IL[l].register_forward_hook(shk(l)) for l in directions]
    try:
        with torch.no_grad():
            g = ivl.generate(
                pixel_values=inp['pixel_values'],
                input_ids=inp['input_ids'],
                attention_mask=inp['attention_mask'],
                do_sample=False, max_new_tokens=80,
            )
        gen = itok.decode(g[0], skip_special_tokens=True)
    except Exception as e:
        gen = f'[gen error {e}]'
    finally:
        for h in hooks:
            h.remove()
    return gen

# ============ Phase 1: direction ============
dir_ckpt = CKPT / 'direction.pkl'
if dir_ckpt.exists():
    d = pickle.load(open(dir_ckpt, 'rb'))
    direction = d['direction']; diag = d['diag']
    print('Loaded cached direction.', flush=True)
else:
    matched = c0f.sample(min(N_DIR, len(c0f)), random_state=SEED).reset_index(drop=True)
    mismatched = mmf.sample(min(N_DIR, len(mmf)), random_state=SEED).reset_index(drop=True)
    m_states = {l: [] for l in CRITICAL_LAYERS}
    mm_states = {l: [] for l in CRITICAL_LAYERS}
    for i, (_, r) in enumerate(matched.iterrows()):
        if i % 50 == 0: print(f'matched {i}/{len(matched)}', flush=True)
        hs = extract_hidden(r['image_path'], r['ret_report'])
        for l in CRITICAL_LAYERS:
            if l in hs: m_states[l].append(hs[l])
    for i, (_, r) in enumerate(mismatched.iterrows()):
        if i % 50 == 0: print(f'mismatched {i}/{len(mismatched)}', flush=True)
        hs = extract_hidden(r['image_path'], r['ret_report'])
        for l in CRITICAL_LAYERS:
            if l in hs: mm_states[l].append(hs[l])
    direction = {}
    diag = {}
    for l in CRITICAL_LAYERS:
        A = torch.stack(m_states[l]).float(); B = torch.stack(mm_states[l]).float()
        mA, mB = A.mean(0), B.mean(0)
        dvec = (mB - mA); dvec = dvec / (dvec.norm() + 1e-9)
        direction[l] = dvec
        diag[l] = {
            'cos_sim_matched_vs_mismatched_mean': float(torch.nn.functional.cosine_similarity(mA.unsqueeze(0), mB.unsqueeze(0)).item()),
            'mean_l2_dist': float((mB - mA).norm().item()),
            'n_matched': int(A.shape[0]), 'n_mismatched': int(B.shape[0]),
        }
    pickle.dump({'direction': direction, 'diag': diag}, open(dir_ckpt, 'wb'))
    print('Direction computed + saved.', flush=True)

print('\n--- Direction diagnostics ---', flush=True)
for l in CRITICAL_LAYERS:
    d = diag[l]
    print(f"  L{l}: cos={d['cos_sim_matched_vs_mismatched_mean']:.4f} L2dist={d['mean_l2_dist']:.4f} n={d['n_matched']}/{d['n_mismatched']}", flush=True)
json.dump(diag, open(OUT / 'direction_diagnostics.json', 'w'), indent=2)

# ============ Phase 2: sweep + FP (resumable) ============
steer_test = mmf.sample(min(N_STEER, len(mmf)), random_state=SEED).reset_index(drop=True)
fp_test = c0f.sample(min(N_FP, len(c0f)), random_state=SEED).reset_index(drop=True)
print(f'Steering test: {len(steer_test)} mismatched, FP: {len(fp_test)} matched', flush=True)

rows_csv = OUT / 'steering_generations_raw.csv'
all_rows = []
done = set()
if rows_csv.exists():
    dfo = pd.read_csv(rows_csv)
    all_rows = dfo.to_dict('records')
    done = set(zip(dfo['uid'].astype(str), dfo['tag'], dfo['alpha'].astype(float)))
    print(f'Resumed {len(all_rows)} rows from CSV.', flush=True)

def run_set(df_, directions, alpha, tag):
    for _, r in df_.iterrows():
        key = (str(r['uid']), tag, float(alpha))
        if key in done:
            continue
        gen = generate_steered(r['image_path'], r['ret_report'], directions=directions, alpha=alpha)
        vn = naive_verdict(gen); vr = robust_verdict(gen)
        all_rows.append({'tag': tag, 'uid': str(r['uid']), 'alpha': alpha, 'naive': vn, 'robust': vr, 'gen': gen})
        done.add(key)
        if len(all_rows) % 10 == 0:
            pd.DataFrame(all_rows).to_csv(rows_csv, index=False)
            print(f'  [checkpoint {len(all_rows)} rows]', flush=True)
    pd.DataFrame(all_rows).to_csv(rows_csv, index=False)

def counts_for(tag, alpha):
    cn = Counter(); cr = Counter()
    for row in all_rows:
        if row['tag'] == tag and float(row['alpha']) == float(alpha):
            cn[row['naive']] += 1; cr[row['robust']] += 1
    return cn, cr

def summarize(tag, alpha):
    cn, cr = counts_for(tag, alpha)
    n = max(sum(cn.values()), 1)
    rn = cn['CONTRADICTS'] / n * 100; rr = cr['CONTRADICTS'] / n * 100
    print(f'{tag} (alpha={alpha}): naive {dict(cn)} | robust {dict(cr)} | naive-CONT {rn:.1f}% robust-CONT {rr:.1f}%', flush=True)
    return rn, rr

# baseline + sweep
run_set(steer_test, None, 0.0, 'sweep')
summarize('sweep', 0.0)
for alpha in ALPHAS:
    run_set(steer_test, direction, alpha, 'sweep')
    summarize('sweep', alpha)
    gc.collect(); torch.cuda.empty_cache()

# false-positive control
run_set(fp_test, direction, FP_ALPHA, 'fp')
summarize('fp', FP_ALPHA)

# ============ final results ============
sweep = {}
for alpha in [0.0] + ALPHAS:
    cn, cr = counts_for('sweep', alpha)
    n = max(sum(cn.values()), 1)
    sweep[alpha] = {'naive': dict(cn), 'robust': dict(cr),
                    'naive_cont': cn['CONTRADICTS']/n*100, 'robust_cont': cr['CONTRADICTS']/n*100}
cnf, crf = counts_for('fp', FP_ALPHA)
results = {
    'model': 'InternVL2-8B',
    'n_direction_samples': diag[CRITICAL_LAYERS[0]]['n_matched'],
    'sweep': {str(a): v for a, v in sweep.items()},
    'fp_alpha': FP_ALPHA,
    'false_positive_naive_counts': dict(cnf),
    'false_positive_robust_counts': dict(crf),
    'false_positive_robust_contradicts_rate': crf['CONTRADICTS']/max(sum(crf.values()),1)*100,
    'direction_diagnostics': diag,
}
json.dump(results, open(OUT / 'caas_results.json', 'w'), indent=2)
print('\n' + json.dumps(results, indent=2), flush=True)

fig, ax = plt.subplots(figsize=(7, 4))
xs = [0.0] + ALPHAS
ax.plot(xs, [sweep[a]['robust_cont'] for a in xs], '-o', color='#e74c3c', lw=2, label='robust CONTRADICTS')
ax.plot(xs, [sweep[a]['naive_cont'] for a in xs], '--s', color='#95a5a6', lw=1.5, label='naive CONTRADICTS')
ax.axhline(sweep[0.0]['robust_cont'], color='gray', ls=':', lw=1)
ax.set_xlabel('Steering strength (alpha)'); ax.set_ylabel('CONTRADICTS rate (%)')
ax.set_title('CAAS: steering the conflict direction (InternVL2-8B)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / 'figures' / 'fig_steering_sweep.png', dpi=300); plt.close()
print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
