"""Check the page's numbers against the Gaussians it ships, without retraining.

Decodes the float16 snapshots in twelve-data.js, renders every camera with the NumPy
rasterizer (gs.py), and compares train / held-out PSNR and the opacity > 0.05 count with
the numbers embedded in seen-from-twelve-sides.html. Needs NumPy and data/gt.npz
(run scene.py first; it needs Pillow only for the contact sheet written after gt.npz).

Usage: cd code && python3 scene.py && python3 check_page.py
"""
import base64, json, os, re, sys
import numpy as np
import gs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOL_DB = 0.05   # float16 storage moves PSNR by a few hundredths of a dB at most

html = open(os.path.join(ROOT, 'seen-from-twelve-sides.html'), encoding='utf-8').read()
META = json.loads(re.search(r'<script id="meta" type="application/json">(.*?)</script>', html, re.S).group(1))
js = open(os.path.join(ROOT, 'twelve-data.js'), encoding='utf-8').read()
blob = base64.b64decode(re.search(r'const BLOB = "([^"]+)"', js).group(1))
N, F, S = META['n'], 14, len(META['steps'])
snaps = np.frombuffer(blob, '<f2').astype(np.float32).reshape(S, N, F)

D = np.load(os.path.join(HERE, 'data', 'gt.npz'))
NT, W, H, FOC = int(D['n_train']), int(D['W']), int(D['H']), float(D['focal'])
cams = [(D['R'][k].astype(np.float32), D['t'][k].astype(np.float32), np.float32(FOC), W, H) for k in range(len(D['R']))]
RGB = D['rgb'].astype(np.float32)

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)

def params(a):
    # page layout per Gaussian: mu(3) log_s(3) q(4, unit) colour(3, activated) opacity(1, activated)
    return dict(mu=a[:, 0:3], log_s=a[:, 3:6], q=a[:, 6:10], c=logit(a[:, 10:13]), o=logit(a[:, 13]))

def psnr(img, ref):
    return float(-10 * np.log10(max(np.mean((img - ref) ** 2), 1e-12)))

worst = 0.0
for k, s in enumerate(META['snaps']):
    P = params(snaps[k])
    p = [psnr(gs.render(P, cams[i], np.zeros(3, np.float32))[0], RGB[i]) for i in range(len(cams))]
    tr, he = float(np.mean(p[:NT])), float(np.mean(p[NT:]))
    alive = int((snaps[k][:, 13] > 0.05).sum())
    d = max(abs(tr - s['train_mean']), abs(he - s['held_mean']))
    worst = max(worst, d)
    print(f"step {s['step']:5d}  train {s['train_mean']:6.2f} -> {tr:6.2f}  held {s['held_mean']:6.2f} -> {he:6.2f}"
          f"  opacity>0.05 {s.get('alive')} -> {alive}", flush=True)

# README: "Counting stone pixels only, step 3,000 scores 31.9 dB ... and 27.2 dB" (stone = any coverage, alpha > 0)
P, A = params(snaps[-1]), D['alpha']
stone = [-10 * np.log10(np.mean((gs.render(P, cams[i], np.zeros(3, np.float32))[0] - RGB[i])[A[i] > 0] ** 2)) for i in range(len(cams))]
st_tr, st_he = float(np.mean(stone[:NT])), float(np.mean(stone[NT:]))
print(f'stone pixels only, step {META["snaps"][-1]["step"]}: train 31.9 -> {st_tr:.2f}  held 27.2 -> {st_he:.2f}')
worst = max(worst, abs(round(st_tr, 1) - 31.9), abs(round(st_he, 1) - 27.2))

print(f'worst PSNR difference {worst:.3f} dB (tolerance {TOL_DB})')
print('PASS' if worst <= TOL_DB else 'FAIL')
sys.exit(0 if worst <= TOL_DB else 1)
