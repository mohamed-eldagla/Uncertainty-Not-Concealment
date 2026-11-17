#!/usr/bin/env python3
"""
Two-Way Forced Choice — InternVL2-8B (SFT), negation-aware

Removes the INCONCLUSIVE escape hatch to test whether the model can *voice* a
contradiction when forced to choose SUPPORTS vs CONTRADICTS.

Fixes vs. old_ref CELL 7 / attempt_3 forced_choice.py:
  - Model is InternVL2-8B (SFT) (matches the CAAS run).
  - NEGATION-AWARE classification: "does not contradict" / "no contradiction"
    / "cannot contradict" -> SUPPORTS (not CONTRADICTS). Only affirmative
    "contradicts" -> CONTRADICTS.
  - Matched C0 control included.
  - Covers C2 + C3 + C4 (per-condition breakdown), not just C3.
  - Raw text saved; all CONTRADICT-containing responses printed for spot-check.
  - Resumable (CSV flushed every 10 rows).

Interpretation:
  - High CONTRADICTS on mismatched + ~0% on matched -> model can voice conflict
    when forced (hedging/concealment).
  - Low CONTRADICTS on mismatched (comparable to matched) -> genuinely cannot
    voice the contradiction (uncertainty, not concealment).
"""
import gc, json, random, re, warnings
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
OUT = Path('/kaggle/working/forced_choice_internvl2_paper'); OUT.mkdir(exist_ok=True, parents=True)
print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu', flush=True)

N_MISMATCH = 150
N_MATCHED = 150
MAX_NEW = 40

# --- negation-aware parser ---
_NEG = re.compile(r'\b(NOT|NO|NEVER|WITHOUT|CANNOT|UNABLE|FAILS?|ABSENT|LACKS?)\b')

def is_negated(prefix):
    return bool(_NEG.search(prefix)) or "N'T" in prefix

def forced_verdict(ans):
    a = (ans or '').strip()
    if not a:
        return 'OTHER', False
    # The model's actual choice is the first token/line. Do not let later
    # option lists ("Support\\nContradict\\nCannot say") override it.
    first = re.match(r'\s*([A-Za-z]+)', a)
    if first:
        w = first.group(1).upper()
        if w.startswith('SUPPORT'):
            return 'SUPPORTS', False
        if w.startswith('CONTRADICT'):
            return 'CONTRADICTS', False
        if w.startswith('CANNOT') or w.startswith('INCONCLUSIVE'):
            return 'OTHER', False
    # Fallback: first label occurrence, with negation handling.
    m = re.search(r'\b(SUPPORTS?|CONTRADICTS?|CANNOT SAY|INCONCLUSIVE)\b', a, re.I)
    if m:
        w = m.group(1).upper()
        if w.startswith('CONTRADICT'):
            if is_negated(a[:m.start()].upper()):
                return 'SUPPORTS', True
            return 'CONTRADICTS', False
        if w.startswith('SUPPORT'):
            return 'SUPPORTS', False
    return 'OTHER', False

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
c0f = c0f[['uid','report_text','image_path']].rename(columns={'report_text':'ret_report'}).reset_index(drop=True)
c0f['ret_report'] = c0f['ret_report'].fillna('').astype(str)
print(f'matched(C0)={len(c0f)} mismatched(C2/C3/C4)={len(mmf)}', flush=True)

# --- Load InternVL2-8B (SFT) ---
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
LOCAL = '/root/models/internvl2_sft_model'
import pathlib
if not pathlib.Path(LOCAL).exists():
    print('Downloading InternVL2-8B ...', flush=True)
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
print('InternVL2-8B loaded.', flush=True)

FORCED_TMPL = ('<|im_start|>user\n<image>\nDoes this X-ray SUPPORT or CONTRADICT the following finding?\n'
               '{finding}\nYou must choose exactly one. Do not say inconclusive.<|im_end|>\n<|im_start|>assistant\n')
tf = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def forced_choice(img_path, report):
    if not isinstance(report, str) or not report.strip():
        return 'OTHER', False, ''
    pr = FORCED_TMPL.format(finding=report)
    img_tok = '<img>' + '<IMG_CONTEXT>' * 256 + '</img>'
    pr = pr.replace('<image>', img_tok)
    inp = itok(pr, return_tensors='pt').to('cuda')
    try:
        inp['pixel_values'] = tf(Image.open(img_path).convert('RGB')).unsqueeze(0).to('cuda').half()
    except Exception:
        return 'OTHER', False, ''
    inp['image_flags'] = torch.tensor([1], dtype=torch.long, device='cuda')
    try:
        with torch.no_grad():
            g = ivl.generate(pixel_values=inp['pixel_values'], input_ids=inp['input_ids'],
                             attention_mask=inp['attention_mask'], do_sample=False, max_new_tokens=MAX_NEW)
        ans = itok.decode(g[0], skip_special_tokens=True)
    except Exception as e:
        ans = f'[gen error {e}]'
    v, neg = forced_verdict(ans)
    return v, neg, ans

# --- resumable run ---
rows_csv = OUT / 'forced_choice_rows.csv'
all_rows = []
done = set()
if rows_csv.exists():
    dfo = pd.read_csv(rows_csv)
    all_rows = dfo.to_dict('records')
    if 'cond' in dfo.columns:
        done = set(zip(dfo['uid'].astype(str), dfo['tag'], dfo['cond'].astype(str)))
    else:
        done = set(zip(dfo['uid'].astype(str), dfo['tag']))
    print(f'Resumed {len(all_rows)} rows.', flush=True)

def run(df_, tag, n):
    sub = df_.sample(min(n, len(df_)), random_state=SEED).reset_index(drop=True)
    for i, (_, r) in enumerate(sub.iterrows()):
        key = (str(r['uid']), tag, str(r.get('cond', 'C0')))
        if key in done:
            continue
        v, neg, ans = forced_choice(r['image_path'], r['ret_report'])
        all_rows.append({'tag': tag, 'cond': r.get('cond', 'C0'), 'uid': str(r['uid']),
                         'verdict': v, 'negated': neg, 'gen': ans})
        done.add(key)
        if i % 10 == 0:
            print(f'{tag} {i}/{len(sub)}', flush=True)
        if len(all_rows) % 10 == 0:
            pd.DataFrame(all_rows).to_csv(rows_csv, index=False)
    pd.DataFrame(all_rows).to_csv(rows_csv, index=False)

print('\n--- Mismatched (C2/C3/C4) forced choice: 100 per condition ---', flush=True)
for _tag, _df in [('C2', c2f), ('C3', c3f), ('C4', c4f)]:
    run(_df.assign(cond=_tag), 'mismatched', 100)
print('--- Matched (C0) forced choice ---', flush=True)
run(c0f, 'matched', N_MATCHED)

# --- results ---
def agg(tag):
    sub = [r for r in all_rows if r['tag'] == tag]
    c = Counter(r['verdict'] for r in sub)
    n = max(sum(c.values()), 1)
    return c, n

mm_c, n_mm = agg('mismatched')
m_c, n_m = agg('matched')
mm_rate = mm_c['CONTRADICTS'] / n_mm * 100
m_rate = m_c['CONTRADICTS'] / n_m * 100
print(f'\nCONTRADICTS rate: mismatched {mm_rate:.1f}% vs matched {m_rate:.1f}%', flush=True)
print(f'Mismatched counts: {dict(mm_c)}', flush=True)
print(f'Matched counts: {dict(m_c)}', flush=True)

# per-condition (mismatched)
per_cond = {}
for cond in ['C2', 'C3', 'C4']:
    sub = [r for r in all_rows if r['tag'] == 'mismatched' and r['cond'] == cond]
    c = Counter(r['verdict'] for r in sub)
    n = max(sum(c.values()), 1)
    per_cond[cond] = {'counts': dict(c), 'n': n, 'contradicts_rate': c['CONTRADICTS'] / n * 100}
    print(f'  {cond}: {dict(c)} CONT={c["CONTRADICTS"]/n*100:.1f}% (n={n})', flush=True)

results = {
    'model': 'InternVL2-8B',
    'mismatched_counts': dict(mm_c),
    'mismatched_contradicts_rate': mm_rate,
    'matched_counts': dict(m_c),
    'matched_contradicts_rate': m_rate,
    'per_condition': per_cond,
    'n_mismatched': n_mm,
    'n_matched': n_m,
}
json.dump(results, open(OUT / 'forced_choice_results.json', 'w'), indent=2)
print('\n' + json.dumps(results, indent=2), flush=True)

# --- spot-check: every response containing CONTRADICT ---
print('\n=== ALL responses containing CONTRADICT (verify negation handling) ===', flush=True)
for r in all_rows:
    if 'CONTRADICT' in (r['gen'] or '').upper():
        print(f"  [{r['tag']} {r['cond']} {r['uid']}] -> {r['verdict']} (negated={r['negated']}): {repr(r['gen'][:160])}", flush=True)

# --- sample OTHER responses ---
print('\n=== sample OTHER (refusals/hedges) ===', flush=True)
others = [r for r in all_rows if r['verdict'] == 'OTHER'][:10]
for r in others:
    print(f"  [{r['tag']} {r['cond']} {r['uid']}]: {repr(r['gen'][:160])}", flush=True)

print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
