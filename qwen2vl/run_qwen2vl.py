#!/usr/bin/env python3
"""
Qwen2-VL-7B Evaluation (corrected conditions, fixed baseline/AEA convention)

Fix: CLS features for C0 AND C2/C3 are extracted in BASELINE mode (same
template), so the probe label ("matching vs mismatching report") is NOT
confounded with the prompt template. AEA mode is used ONLY to generate the
behavioral verdicts.
"""
import gc, json, random, warnings, pickle, re
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
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path('/kaggle/working/qwen2vl_output_paper'); OUT.mkdir(exist_ok=True, parents=True)
(OUT / 'figures').mkdir(exist_ok=True, parents=True)
(OUT / 'tables').mkdir(exist_ok=True, parents=True)
CKPT = Path('/kaggle/working/checkpoints_qwen2vl_paper'); CKPT.mkdir(exist_ok=True, parents=True)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)

# --- Load corrected pipeline data ---
PREV = next((p for p in [
    Path('/kaggle/working/aea-xai-pipeline'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/corrected-aea-xai-pipeline-output'),
    Path('/kaggle/input/corrected-aea-xai-pipeline-output'),
] if p.is_dir()), None)
if PREV is None:
    raise FileNotFoundError('corrected pipeline output not found')
splits = pd.read_csv(PREV / 'splits.csv')
test = splits[splits['split'].eq('test')].copy()
c2f = pd.read_csv(PREV / 'C2.csv'); c3f = pd.read_csv(PREV / 'C3.csv'); c4f = pd.read_csv(PREV / 'C4.csv')

IU = next((p for p in [
    Path('/kaggle/input/datasets/raddar/chest-xrays-indiana-university'),
    Path('/kaggle/input/datasets/SET_PATH_HERE/chest-xrays-indiana-university'),
    Path('/kaggle/input/chest-xrays-indiana-university'),
] if p.is_dir()), None)
if IU is None:
    raise FileNotFoundError('IU dataset not found')
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
print(f'C0={len(c0f)} C2={len(c2f)} C3={len(c3f)} C4={len(c4f)}', flush=True)

# --- Load Qwen2-VL-7B-Instruct ---
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
QID = 'Qwen/Qwen2-VL-7B-Instruct'
qw = Qwen2VLForConditionalGeneration.from_pretrained(
    QID,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    torch_dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True,
)
qp = AutoProcessor.from_pretrained(QID, min_pixels=256*28*28, max_pixels=1280*28*28)
QL = qw.model.language_model.layers
QN = len(QL)
print(f'Qwen2-VL loaded. {QN} layers.', flush=True)

QP_AEA_TEXT = ("Retrieved record suggests: {finding}.\n"
    "Does this X-ray SUPPORT, CONTRADICT, or is INCONCLUSIVE about '{finding}'?\n"
    "Check: (a) lung opacification (b) costophrenic angles (c) cardiac silhouette (d) pleural space.\n"
    "Reply ONLY:\nVISUAL EVIDENCE: [SUPPORTS/CONTRADICTS/INCONCLUSIVE]\nRATIONALE: [one sentence]")
QP_BASE_TEXT = ("Retrieved record: {report}\nDiagnosis based on image and record?")

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

def qpass(img_path, report, mode='baseline'):
    if not isinstance(report, str) or not report.strip():
        return None, None, None
    img = Image.open(img_path).convert('RGB')
    question = QP_AEA_TEXT.format(finding=report) if mode == 'aea' else QP_BASE_TEXT.format(report=report)
    messages = [{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':question}]}]
    text = qp.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = qp(text=[text], images=[img], padding=True, return_tensors='pt').to(qw.device)
    cap = {}
    def capture_hidden(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cap['h'] = hidden.detach().float()
    hook = QL[QN - 2].register_forward_hook(capture_hidden)
    gen = None
    try:
        if mode == 'aea':
            with torch.no_grad():
                generated = qw.generate(**inp, max_new_tokens=80, do_sample=False)
            prompt_len = inp['input_ids'].shape[1]
            gen = qp.batch_decode(generated[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        else:
            with torch.no_grad():
                qw(**inp)
    finally:
        hook.remove()
    cls = cap['h'][0, -1].cpu().numpy().astype('float32') if 'h' in cap else None
    torch.cuda.empty_cache()
    return cls, None, gen

print('Qwen helpers ready.', flush=True)

# --- Extraction (checkpointed, resumable per condition) ---
ckpt_path = CKPT / 'cls_aea.pkl'
if ckpt_path.exists():
    d = pickle.load(open(ckpt_path, 'rb'))
    cl_f, cl_l, cl_g, ar = d['cl_f'], d['cl_l'], d['cl_g'], d['ar']
    done = {r['cond'] for r in ar}
    c0_done = len(cl_f) > 0
    print(f'Checkpoint loaded: CLS {len(cl_f)}, AEA {len(ar)}, done={sorted(done)}, c0_done={c0_done}', flush=True)
else:
    cl_f, cl_l, cl_g, ar = [], [], [], []
    done = set()
    c0_done = False

if not c0_done:
    # C0: baseline CLS (label 0)
    for i, (_, r) in enumerate(c0f.iterrows()):
        if i % 50 == 0: print(f'C0 {i}/{len(c0f)}', flush=True)
        c, _, _ = qpass(r['image_path'], r['ret_report'], mode='baseline')
        if c is not None:
            cl_f.append(c); cl_l.append(0); cl_g.append(str(r['uid']))
    pickle.dump({'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g, 'ar': ar}, open(ckpt_path, 'wb'))
    print('  Checkpoint after C0', flush=True)

# C2/C3/C4: baseline CLS (label 1 for C2/C3) + AEA verdicts
for tag, df_ in [('C2', c2f), ('C3', c3f), ('C4', c4f)]:
    if tag in done:
        print(f'  {tag} already in checkpoint; skipping', flush=True)
        continue
    for i, (_, r) in enumerate(df_.iterrows()):
        if i % 50 == 0: print(f'{tag} {i}/{len(df_)}', flush=True)
        # baseline CLS (same template as C0)
        cb, _, _ = qpass(r['image_path'], r['ret_report'], mode='baseline')
        # AEA verdict (same input)
        _, _, g = qpass(r['image_path'], r['ret_report'], mode='aea')
        v = robust_verdict(g)
        ar.append({'cond': tag, 'uid': r['uid'], 'verdict': v, 'gen': g})
        if cb is not None and tag in ('C2', 'C3'):
            cl_f.append(cb); cl_l.append(1); cl_g.append(str(r['uid']))
    pickle.dump({'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g, 'ar': ar}, open(ckpt_path, 'wb'))
    print(f'  Checkpoint after {tag}', flush=True)
    done.add(tag)

pickle.dump({'cl_f': cl_f, 'cl_l': cl_l, 'cl_g': cl_g, 'ar': ar}, open(ckpt_path, 'wb'))
print(f'Checkpoint saved: {len(cl_f)} CLS rows', flush=True)

X = np.vstack(cl_f).astype('float32'); y = np.array(cl_l); g = np.array(cl_g)
np.savez_compressed(OUT / 'cls_features.npz', X=X, y=y, groups=g)
ad = pd.DataFrame(ar, columns=['cond', 'uid', 'verdict', 'gen'])
ad.to_csv(OUT / 'aea_verdicts.csv', index=False)
print('Verdicts:'); print(ad['verdict'].value_counts().to_string())
print('CLS:', X.shape, 'groups:', len(np.unique(g)), flush=True)

# --- CLS probing AUC (nested grouped) ---
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
print(f'Qwen2-VL CLS AUC={auc:.4f} +/- {std:.4f} folds={folds}', flush=True)

# --- Verdict distribution figure ---
fig, ax = plt.subplots(figsize=(8, 5))
cv = ['C2','C3','C4']
vd = [float((ad[ad['cond'].eq(c)]['verdict'].eq('CONTRADICTS')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
sp = [float((ad[ad['cond'].eq(c)]['verdict'].eq('SUPPORTS')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
ic = [float((ad[ad['cond'].eq(c)]['verdict'].eq('INCONCLUSIVE')).mean()*100) if len(ad[ad['cond'].eq(c)]) else 0 for c in cv]
x = np.arange(len(cv))
ax.bar(x, vd, 0.25, label='CONTRADICTS', color='#e74c3c')
ax.bar(x, sp, 0.25, bottom=vd, label='SUPPORTS', color='#2ecc71')
ax.bar(x, ic, 0.25, bottom=np.array(vd)+np.array(sp), label='INCONCLUSIVE', color='#f39c12')
ax.set_xticks(x); ax.set_xticklabels(cv); ax.set_ylabel('%'); ax.set_title('Qwen2-VL Verdict Distribution')
ax.legend(fontsize=8); ax.set_ylim(0, 100)
plt.tight_layout(); plt.savefig(OUT / 'figures' / 'fig_verdict_dist.png', dpi=300); plt.close()

summary = {'model': 'Qwen2-VL-7B', 'cls_auc': float(auc), 'cls_auc_std': float(std), 'cls_folds': folds,
           'verdicts': ad['verdict'].value_counts().to_dict()}
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=2)
print(json.dumps(summary, indent=2), flush=True)
del qw; gc.collect(); torch.cuda.empty_cache()
print('Done.', flush=True)
