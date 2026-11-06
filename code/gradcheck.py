"""Central-difference check of every hand-derived gradient in gs.py on a tiny case (float64).

6 overlapping anisotropic Gaussians, a 20x20 image, a random linear loss L = sum(img * K), a grey background.
Run twice: with culling off (the smooth function) and with the training settings (tiles, 3-sigma, a<1/255).
"""
import numpy as np, json, sys, os
import gs
from scene import look_at, cam_pos

def make(seed=0, N=6):
    r = np.random.default_rng(seed)
    return dict(mu=r.normal(0, 0.25, (N, 3)), log_s=np.log(r.uniform(0.08, 0.3, (N, 3))),
                q=r.normal(0, 1, (N, 4)), c=r.normal(0, 1, (N, 3)), o=r.normal(0.5, 1, N))

def check(cull, seed=0):
    P = make(seed)
    R, t = look_at(cam_pos(30, 20) / 3.15 * 2.2)
    W = H = 20; f = 0.5 * W / np.tan(np.radians(38) / 2)
    cam = (R, t, f, W, H)
    bg = np.array([0.3, 0.5, 0.7])
    K = np.random.default_rng(seed + 1).normal(0, 1, (H, W, 3))
    lg = lambda img: ((img * K).sum(), K)
    loss, img, g, _ = gs.forward_backward(P, cam, bg, lg, cull=cull)
    out = {}
    eps = 1e-6
    for k in P:
        num = np.zeros_like(P[k]); it = np.nditer(P[k], flags=['multi_index'])
        for _ in it:
            i = it.multi_index
            Pp = {kk: v.copy() for kk, v in P.items()}; Pm = {kk: v.copy() for kk, v in P.items()}
            Pp[k][i] += eps; Pm[k][i] -= eps
            lp = (gs.rasterize(gs.activate(Pp), gs.project(gs.activate(Pp), R, t, f, W, H, cull), W, H, bg, cull=cull)[0] * K).sum()
            lm = (gs.rasterize(gs.activate(Pm), gs.project(gs.activate(Pm), R, t, f, W, H, cull), W, H, bg, cull=cull)[0] * K).sum()
            num[i] = (lp - lm) / (2 * eps)
        rel = np.abs(num - g[k]).max() / max(1e-12, np.abs(num).max())
        out[k] = float(rel)
    return out, float(img.min()), float(img.max())

if __name__ == '__main__':
    res = {}
    for cull in (False, True):
        for seed in (0, 1):
            r, lo, hi = check(cull, seed)
            res[f'cull={cull},seed={seed}'] = r
            print(f'cull={cull!s:5} seed={seed}  ' + '  '.join(f'{k}:{v:.1e}' for k, v in r.items()))
    worst = max(v for r in res.values() for v in r.values())
    print('worst max relative error:', f'{worst:.2e}', 'PASS' if worst < 1e-5 else 'FAIL')
    json.dump(dict(results=res, worst=worst), open(os.path.join(os.path.dirname(__file__), 'data', 'gradcheck.json'), 'w'), indent=1)
