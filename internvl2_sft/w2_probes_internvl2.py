#!/usr/bin/env python3
"""
W2 — Simpler probe baselines (InternVL2-8B SFT), no MDI.

Establishes whether SIMPLER signals than our CLS probe (AUC ~0.80) also separate
matched (C0) vs mismatched (C2/C3/C4) pairs:
  1. CLIP cosine similarity (BiomedCLIP image-text).
  2. Raw output entropy (model's own next-token distribution).
  3. Cross-modal attention (text->visual, last layer) — best-effort.

For each probe we report matched vs mismatched mean + AUC (threshold discriminator)
+ Mann-Whitney U p-value, so each simpler probe can be compared to the CLS AUC.
"""
import gc, json, random, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = 'cuda:0'
OUT = Path('/kaggle/working/w2_probes_paper'); OUT.mkdir(exist_ok=True, parents=True)

N_PER = 200   # matched + mismatched samples

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

# --- Load InternVL2-8B (SFT) on GPU 1 ---
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
print(f'InternVL2-8B loaded on {DEV}.', flush=True)

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

def vlm_forward(img_path, report):
    """Returns (output_entropy, cross_modal_attention, cls_feature)."""
    inp = build_inp(img_path, report)
    if inp is None:
        return None, None, None
    ids = inp['input_ids'][0]
    vis_pos = (ids == ivl.img_context_token_id).nonzero(as_tuple=True)[0]
    txt_pos = (ids != ivl.img_context_token_id).nonzero(as_tuple=True)[0]
    try:
        with torch.no_grad():
            out = ivl(**inp, output_hidden_states=True, output_attentions=True, return_dict=True)
    except Exception:
        with torch.no_grad():
            out = ivl(**inp, output_hidden_states=True, return_dict=True)
        out = out
    # output entropy
    logits = out.logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    ent = -(p * (p + 1e-9).log()).sum().item()
    # cross-modal attention (best-effort)
    attn_score = None
    attns = getattr(out, 'attentions', None)
    if attns is not None and len(attns) > 0:
        try:
            ma = attns[-1][0].mean(dim=0)  # (seq, seq)
            cross = ma[txt_pos.cpu()][:, vis_pos.cpu()].mean().item()
            self_t = ma[txt_pos.cpu()][:, txt_pos.cpu()].mean().item()
            attn_score = cross / (self_t + 1e-9)
        except Exception:
            attn_score = None
    # cls feature (layer 30 last token)
    cls = out.hidden_states[-2][0, -1].float().cpu().numpy().astype('float32')
    return ent, attn_score, cls

# --- Load BiomedCLIP on GPU 1 ---
print('Loading BiomedCLIP ...', flush=True)
import open_clip
BIO = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
clip_model, _, clip_pre = open_clip.create_model_and_transforms(BIO)
clip_tok = open_clip.get_tokenizer(BIO)
clip_model = clip_model.to(DEV).eval()
print('BiomedCLIP loaded.', flush=True)

def clip_cosine(img_path, report):
    try:
        # Do not silently truncate at BiomedCLIP's 256-token encoder limit.
        ids = clip_tok([report], context_length=512)[0]
        if int((ids != 0).sum()) > 256:
            return None
        img = clip_pre(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEV)
        txt = clip_tok([report]).to(DEV)
        with torch.no_grad():
            if_ = clip_model.encode_image(img); if_ = if_ / if_.norm(dim=-1, keepdim=True)
            tf_ = clip_model.encode_text(txt); tf_ = tf_ / tf_.norm(dim=-1, keepdim=True)
        return float((if_ @ tf_.T).item())
    except Exception:
        return None

# --- Run ---
matched = c0f.sample(min(N_PER, len(c0f)), random_state=SEED).reset_index(drop=True)
mismatched = mmf.sample(min(N_PER, len(mmf)), random_state=SEED).reset_index(drop=True)

rows = []
for tag, df_, repcol in [('matched', matched, 'report'), ('mismatched', mismatched, 'report')]:
    for i, (_, r) in enumerate(df_.iterrows()):
        if i % 50 == 0: print(f'{tag} {i}/{len(df_)}', flush=True)
        ent, attn, cls = vlm_forward(r['image_path'], r[repcol])
        cos = clip_cosine(r['image_path'], r[repcol])
        rows.append({'tag': tag, 'uid': r['uid'], 'clip_cosine': cos,
                     'output_entropy': ent, 'attention': attn, 'cls': cls})
        if i % 100 == 0 and i > 0:
            pd.DataFrame(rows).to_csv(OUT / 'w2_rows.csv', index=False)
pd.DataFrame(rows).to_csv(OUT / 'w2_rows.csv', index=False)
print(f'Collected {len(rows)} rows.', flush=True)

# --- Summarize ---
df = pd.DataFrame(rows)
results = {}
for probe in ['clip_cosine', 'output_entropy', 'attention']:
    m = df[df['tag'] == 'matched'][probe].dropna()
    mm = df[df['tag'] == 'mismatched'][probe].dropna()
    if len(m) == 0 or len(mm) == 0:
        results[probe] = {'available': False}
        print(f'{probe}: not available', flush=True)
        continue
    X = np.concatenate([m.values, mm.values])
    y = np.array([0] * len(m) + [1] * len(mm))
    auc = roc_auc_score(y, X)
    u = mannwhitneyu(mm.values, m.values, alternative='two-sided')
    results[probe] = {'available': True, 'matched_mean': float(m.mean()), 'mismatched_mean': float(mm.mean()),
                      'auc': float(auc), 'mannwhitney_p': float(u.pvalue), 'n_matched': int(len(m)), 'n_mismatched': int(len(mm))}
    print(f'{probe}: matched={m.mean():.4f} mismatched={mm.mean():.4f} AUC={auc:.3f} p={u.pvalue:.4f}', flush=True)

json.dump({'model': 'InternVL2-8B', 'reference_cls_auc': 0.798, 'probes': results}, open(OUT / 'w2_results.json', 'w'), indent=2)
print('\n' + json.dumps(results, indent=2), flush=True)
print('Done.', flush=True)
del ivl; gc.collect(); torch.cuda.empty_cache()
