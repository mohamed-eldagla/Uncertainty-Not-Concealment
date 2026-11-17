#!/usr/bin/env python3
"""
InternVL2-8B Evaluation (Corrected Conditions) — checkpointed
Behavioral generation + CLS probing, No MDI, fixed paths and gna syntax
"""
import gc, json, random, pickle, warnings, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path('/kaggle/working/internvl2_sft_output_paper'); OUT.mkdir(exist_ok=True, parents=True)
(OUT / 'figures').mkdir(exist_ok=True, parents=True)
(OUT / 'tables').mkdir(exist_ok=True, parents=True)
CKPT = Path('/kaggle/working/checkpoints_internvl2_sft_paper'); CKPT.mkdir(exist_ok=True, parents=True)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)
print(f'Checkpoint dir: {CKPT}', flush=True)

# --- Load corrected pipeline data ---
PREV_CANDIDATES = [
    Path('/kaggle/working/aea-xai-pipeline'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/corrected-aea-xai-pipeline-output'),
    Path("/data/corrected-aea-xai-pipeline-output"),
    Path('/kaggle/input/corrected-aea-xai-pipeline-output'),
]
PREV = next((p for p in PREV_CANDIDATES if p.is_dir()), None)
if PREV is None:
    raise FileNotFoundError(f'corrected pipeline output not found, tried: {PREV_CANDIDATES}')
print(f'PREV={PREV}', flush=True)
splits = pd.read_csv(PREV / 'splits.csv'); test = splits[splits['split'].eq('test')].copy()
c2f = pd.read_csv(PREV / 'C2.csv'); c3f = pd.read_csv(PREV / 'C3.csv'); c4f = pd.read_csv(PREV / 'C4.csv')

IU_CANDIDATES = [
    Path('/kaggle/input/datasets/raddar/chest-xrays-indiana-university'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/chest-xrays-indiana-university'),
    Path('/kaggle/input/chest-xrays-indiana-university'),
]
IU = next((p for p in IU_CANDIDATES if p.is_dir()), None)
if IU is None:
    raise FileNotFoundError(f'IU not found')
print(f'IU={IU}', flush=True)
proj = pd.read_csv(IU / 'indiana_projections.csv')
img_dir = next((p for p in [IU/'images'/'images_normalized', IU/'images', IU] if p.is_dir()), IU)
print(f'img_dir={img_dir}', flush=True)
# efficient mapping
all_img_files = list(img_dir.rglob('*'))
fname_to_path = {p.name: p for p in all_img_files if p.is_file()}
uid_to_path = {}
for _, r in proj[proj['projection'].eq('Frontal')].iterrows():
    fn = str(r['filename'])
    if fn in fname_to_path:
        uid_to_path[r['uid']] = fname_to_path[fn]
print(f'Mapped {len(uid_to_path)} frontal', flush=True)

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
# ensure c0 ret_report is str
c0f['ret_report'] = c0f['ret_report'].fillna('').astype(str)
print(f'C0={len(c0f)} C2={len(c2f)} C3={len(c3f)} C4={len(c4f)}', flush=True)

# --- Load InternVL2-8B (SFT) ---
print('Loading InternVL2-8B (SFT) ...', flush=True)
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from huggingface_hub import snapshot_download
MID = 'OpenGVLab/InternVL2-8B'
# Download to local disk (so we can patch the meta-tensor .item() bug)
LOCAL = '/root/models/internvl2_sft_model'
import pathlib
if not pathlib.Path(LOCAL).exists():
    print(f'Downloading {MID} to {LOCAL} ...', flush=True)
    snapshot_download(repo_id=MID, local_dir=LOCAL)
    print('Download done.', flush=True)
# Patch meta-tensor bug in custom ViT code (needed for 8-bit/init_empty_weights)
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
# InternVL2 requires img_context_token_id set (normally done by chat()); set it manually
ivl.img_context_token_id = itok.convert_tokens_to_ids('<IMG_CONTEXT>')
# layers
try:
    IL = ivl.language_model.model.layers
except AttributeError:
    IL = ivl.language_model.layers
IN = len(IL); print(f'InternVL2-8B loaded. {IN} layers.', flush=True)
IP_AEA = ('<|im_start|>user\n<image>\nRetrieved record suggests: {finding}.\n'
    'Does this X-ray SUPPORT, CONTRADICT, or is INCONCLUSIVE about \'{finding}\'?\n'
    'Check: (a) lung opacification (b) costophrenic angles (c) cardiac silhouette (d) pleural space.\n'
    'Reply ONLY:\nVISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]\nRATIONALE: [one sentence]<|im_end|>\n<|im_start|>assistant\n')
IP_BASE = ('<|im_start|>user\n<image>\nRetrieved record: {report}\nDiagnosis based on image and record?<|im_end|>\n<|im_start|>assistant\n')
tf = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def robust_verdict(text):
    t = (text or '').strip().upper()
    first_line = t.splitlines()[0] if t else ''
    if all(x in first_line for x in ('SUPPORTS', 'CONTRADICTS', 'INCONCLUSIVE')) and ('/' in first_line or '|' in first_line):
        return 'INVALID'
    m = re.search(r'VISUAL EVIDENCE\s*:?\s*\[?\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)', t)
    if m:
        return m.group(1)
    m = re.match(r'\s*(SUPPORTS|CONTRADICTS|INCONCLUSIVE)\b', t)
    if m:
        return m.group(1)
    return 'INVALID'

def ipass(img_path, report, mode='baseline'):
    if not isinstance(report, str) or not report.strip():
        return None, None, None
    pr = IP_AEA.format(finding=report) if mode == 'aea' else IP_BASE.format(report=str(report))
    img_tok = '<img>' + '<IMG_CONTEXT>' * 256 + '</img>'
    pr = pr.replace('<image>', img_tok)
    inp = itok(pr, return_tensors='pt').to('cuda')
    try:
        inp['pixel_values'] = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to('cuda').half()
    except Exception:
        return None, None, None
    inp['image_flags'] = torch.tensor([1], dtype=torch.long, device='cuda')
    gen = None
    cls = None
    # CLS via forward with output_hidden_states
    try:
        with torch.no_grad():
            out = ivl(**inp, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        cls = hs[-2][0, -1].float().cpu().numpy().astype('float32')
    except Exception as e:
        print(f'[ipass cls error {e}]', flush=True)
        cls = None
    if mode == 'aea':
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
    return cls, None, gen

print('InternVL2 helpers ready.', flush=True)

# --- Extract CLS + AEA generation (checkpointed, resumable per condition) ---
ckpt_path = CKPT / 'cls_aea.pkl'
if ckpt_path.exists():
    print(f'Loading checkpoint {ckpt_path} ...', flush=True)
    d = pickle.load(open(ckpt_path, 'rb'))
    fr, ar, cl_f, cl_l, cl_g = d['fr'], d['ar'], d['cl_f'], d['cl_l'], d['cl_g']
    done = {r['cond'] for r in ar}
    c0_done = len(cl_f) > 0
    print(f'  Loaded: CLS {len(cl_f)} rows, AEA {len(ar)}, done={sorted(done)}, c0_done={c0_done}', flush=True)
else:
    fr = []; ar = []; cl_f, cl_l, cl_g = [], [], []
    done = set()
    c0_done = False

if not c0_done:
    # C0
    for i, (_, r) in enumerate(c0f.iterrows()):
        if i % 50 == 0:
            print(f'C0 {i}/{len(c0f)}', flush=True)
        c, _, _ = ipass(r['image_path'], r['ret_report'])
        if c is not None:
            cl_f.append(c); cl_l.append(0); cl_g.append(str(r['uid']))
        if i % 100 == 0 and i>0:
            # intermediate checkpoint
            pickle.dump({'fr': fr, 'ar': ar, 'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g}, open(ckpt_path, 'wb'))
    pickle.dump({'fr': fr, 'ar': ar, 'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g}, open(ckpt_path, 'wb'))
    print('  Checkpoint after C0', flush=True)

# C2/C3/C4: baseline CLS (same template as C0) + AEA verdicts
for tag, df_ in [('C2', c2f), ('C3', c3f), ('C4', c4f)]:
    if tag in done:
        print(f'  {tag} already in checkpoint; skipping', flush=True)
        continue
    for i, (_, r) in enumerate(df_.iterrows()):
        if i % 50 == 0:
            print(f'{tag} {i}/{len(df_)}', flush=True)
        # baseline pass for CLS (same template as C0, no confound)
        cb, _, _ = ipass(r['image_path'], r['ret_report'], mode='baseline')
        # AEA pass for behavioral verdict (same input)
        _, _, g = ipass(r['image_path'], r['ret_report'], mode='aea')
        v = robust_verdict(g)
        ar.append({'cond': tag, 'uid': r['uid'], 'verdict': v, 'gen': g})
        if cb is not None and tag in ('C2', 'C3'):
            fr.append({'uid': str(r['uid']), 'cond': tag, 'cls': cb})
            cl_f.append(cb); cl_l.append(1); cl_g.append(str(r['uid']))
    # checkpoint after each condition
    pickle.dump({'fr': fr, 'ar': ar, 'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g}, open(ckpt_path, 'wb'))
    print(f'  Checkpoint after {tag}', flush=True)
    done.add(tag)
pickle.dump({'fr': fr, 'ar': ar, 'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g}, open(ckpt_path, 'wb'))
print(f'Checkpoint saved: {len(cl_f)} CLS rows', flush=True)

if len(cl_f) == 0:
    print("No CLS features found (all ipass returned None), creating empty summary", flush=True)
    ad = pd.DataFrame(ar) if len(ar) else pd.DataFrame(columns=['cond','uid','verdict','gen'])
    if len(ad): ad.to_csv(OUT / 'aea_verdicts.csv', index=False)
    summary = {'model': 'InternVL2-8B', 'cls_auc': None, 'cls_auc_std': None, 'verdicts': ad['verdict'].value_counts().to_dict() if len(ad) else {}}
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print("Done (empty).", flush=True)
    import sys; sys.exit(0)
X = np.vstack(cl_f).astype('float32'); y = np.array(cl_l); g = np.array(cl_g)
np.savez_compressed(OUT / 'cls_features.npz', X=X, y=y, groups=g)
ad = pd.DataFrame(ar, columns=['cond', 'uid', 'verdict', 'gen']); ad.to_csv(OUT / 'aea_verdicts.csv', index=False)
print('Verdicts:'); print(ad['verdict'].value_counts().to_string())
print('CLS:', X.shape, 'groups:', len(np.unique(g)), flush=True)

# --- CLS probing AUC (fixed mk syntax) ---
def gna(X, y, g):
    o = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    cfg = [(f'raw_C{c:g}', c, None) for c in [1e-5,1e-4,1e-3,1e-2,1e-1,1,10]] + [(f'pca{n}_C{c:g}', c, n) for n in [16,32,64,128] for c in [1e-3,1e-2,1e-1,1]]
    def mk(C, n_comp):
        steps = [('scale', StandardScaler())]
        if n_comp is not None:
            steps.append(('pca', PCA(n_components=n_comp, random_state=SEED)))
        steps.append(('clf', LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=4000, random_state=SEED)))
        return Pipeline(steps)
    fa = []
    for fold, (tr, te) in enumerate(o.split(X, y, g), 1):
        inn = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED+fold)
        isp = list(inn.split(X[tr], y[tr], g[tr]))
        cd = []
        for nm, Cval, n_comp in cfg:
            if n_comp is not None and n_comp > min(X.shape[1], min(len(a) for a, _ in isp)):
                continue
            try:
                aa = [roc_auc_score(y[tr][iv], mk(Cval, n_comp).fit(X[tr][it], y[tr][it]).predict_proba(X[tr][iv])[:,1]) for it, iv in isp]
                cd.append((nm, Cval, n_comp, float(np.mean(aa))))
            except Exception:
                continue
        if not cd:
            continue
        b = max(cd, key=lambda r: r[3])
        cl = mk(b[1], b[2]); cl.fit(X[tr], y[tr])
        fa.append(roc_auc_score(y[te], cl.predict_proba(X[te])[:,1]))
    return float(np.mean(fa)), float(np.std(fa)), fa

auc, std, folds = gna(X, y, g)
print(f'InternVL2 CLS AUC={auc:.4f} +/- {std:.4f} folds={folds}', flush=True)

# --- Figures ---
fig, ax = plt.subplots(figsize=(8, 5))
cv = ['C2','C3','C4']
vd = [float((ad[ad['cond'].eq(c)]['verdict'].eq('CONTRADICTS')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
sp = [float((ad[ad['cond'].eq(c)]['verdict'].eq('SUPPORTS')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
ic = [float((ad[ad['cond'].eq(c)]['verdict'].eq('INCONCLUSIVE')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
x = np.arange(len(cv))
ax.bar(x, vd, 0.25, label='CONTRADICTS', color='#e74c3c')
ax.bar(x, sp, 0.25, bottom=vd, label='SUPPORTS', color='#2ecc71')
ax.bar(x, ic, 0.25, bottom=np.array(vd)+np.array(sp), label='INCONCLUSIVE', color='#f39c12')
ax.set_xticks(x); ax.set_xticklabels(cv); ax.set_ylabel('%'); ax.set_title('InternVL2-8B Verdict Distribution (Corrected)')
ax.legend(fontsize=8); ax.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(OUT / 'figures' / 'fig_verdict_dist.png', dpi=300); plt.close()
print('Figure saved', flush=True)

summary = {'model': 'InternVL2-8B', 'cls_auc': float(auc), 'cls_auc_std': float(std), 'cls_folds': folds, 'verdicts': ad['verdict'].value_counts().to_dict()}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2), flush=True)
print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
