"""Fit N Gaussians to the 12 training photographs. Resumable: ckpt/latest.npz is written every CKPT steps.

Usage: python3 train.py [--steps 3000] [--n 4000] [--fresh]
Writes data/log.json (every number on the page comes from here) and data/snaps/step_XXXXX.npz.
"""
import numpy as np, json, os, time, argparse
import gs

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--n', type=int, default=4000)
ap.add_argument('--fresh', action='store_true')
ap.add_argument('--bench', type=int, default=0)
ap.add_argument('--reset', type=int, default=0, help='3DGS-style opacity reset at this step: every opacity is capped at 0.01')
ap.add_argument('--prune', type=float, default=0.0, help='3DGS-style prune: every PRUNE_EVERY steps from PRUNE_FROM, Gaussians below this opacity are switched off for good')
args = ap.parse_args()

D = np.load(os.path.join(HERE, 'data', 'gt.npz'))
NT = int(D['n_train']); W = int(D['W']); H = int(D['H']); F = float(D['focal'])
cams = [(D['R'][k].astype(np.float32), D['t'][k].astype(np.float32), np.float32(F), W, H) for k in range(len(D['R']))]
RGB = D['rgb'].astype(np.float32); ALPHA = D['alpha'].astype(np.float32)
TRAIN, HELD = list(range(NT)), list(range(NT, len(cams)))

SNAPS = [0, 10, 25, 50, 100, 200, 350, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
CKPT = 250
LR = dict(mu=(4e-3, 4e-5), log_s=(1e-2, 2e-3), q=(4e-3, 1e-3), c=(2.5e-2, 5e-3), o=(4e-2, 1e-2))   # start, end (exp decay)
B1, B2, EPS = 0.9, 0.999, 1e-15
LOG_S_MAX = np.log(0.4)
PRUNE_FROM, PRUNE_EVERY, PRUNE_UNTIL = 300, 300, 2400
OFF = -14.0   # logit of a pruned Gaussian: opacity 8e-7, below the 1/255 cut everywhere, so it gets no gradient

def init(n, seed=0):
    r = np.random.default_rng(seed)
    u = r.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    mu = u * 1.3 * r.random((n, 1)) ** (1 / 3)               # uniform in a ball of radius 1.3
    return dict(mu=mu.astype(np.float32),
                log_s=np.full((n, 3), np.log(0.06), np.float32),
                q=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)) + r.normal(0, 0.3, (n, 4)).astype(np.float32),
                c=r.normal(0, 0.25, (n, 3)).astype(np.float32),      # ~ grey 0.5 +- a little
                o=np.full(n, np.log(0.1 / 0.9), np.float32))          # opacity 0.1, as in 3DGS

def psnr(mse): return float(-10 * np.log10(max(mse, 1e-12)))

def evaluate(P, ids):
    out = []
    for k in ids:
        img, _ = gs.render(P, cams[k], np.zeros(3, np.float32))
        out.append(psnr(np.mean((img - RGB[k]) ** 2)))
    return out

def snapshot(P, step):
    A = gs.activate(P)
    os.makedirs(os.path.join(HERE, 'data', 'snaps'), exist_ok=True)
    np.savez(os.path.join(HERE, 'data', 'snaps', f'step_{step:05d}.npz'), mu=A['mu'], log_s=P['log_s'], q=A['qn'], c=A['c'], o=A['o'])

ck = os.path.join(HERE, 'ckpt', 'latest.npz'); logp = os.path.join(HERE, 'data', 'log.json')
if os.path.exists(ck) and not args.fresh and not args.bench:
    Z = np.load(ck, allow_pickle=True)
    P = {k: Z['P_' + k] for k in LR}; m1 = {k: Z['m_' + k] for k in LR}; m2 = {k: Z['v_' + k] for k in LR}
    step0 = int(Z['step']); rng = np.random.default_rng(); rng.bit_generator.state = Z['rng'].item()
    log = json.load(open(logp))
    log['snaps'] = [s for s in log['snaps'] if s['step'] < step0]; log['loss'] = [l for l in log['loss'] if l[0] < step0]
    print('resumed at step', step0)
else:
    P = init(args.n); m1 = {k: np.zeros_like(v) for k, v in P.items()}; m2 = {k: np.zeros_like(v) for k, v in P.items()}
    step0 = 0; rng = np.random.default_rng(1)
    log = dict(config=dict(n_gaussians=args.n, steps=args.steps, image=W, n_train=NT, n_held=len(HELD), lr=LR,
                           loss='L1 on a random solid background', snaps=SNAPS, prune=args.prune, prune_schedule=[PRUNE_FROM, PRUNE_EVERY, PRUNE_UNTIL] if args.prune else None, opacity_reset=args.reset or None), loss=[], snaps=[], seconds=0.0)

def l1(target):
    def f(img):
        d = img - target
        return float(np.abs(d).mean()), (np.sign(d) / d.size).astype(np.float32)
    return f

t_start = time.time() - log['seconds']
t_last = time.time()
run_loss = []
for step in range(step0, args.steps + 1):
    if step in SNAPS or step == args.steps:
        if not args.bench:
            tr, ho = evaluate(P, TRAIN), evaluate(P, HELD)
            alive = int((gs.sigmoid(P['o']) > 0.05).sum())
            log['snaps'].append(dict(step=step, train_psnr=tr, held_psnr=ho, train_mean=float(np.mean(tr)), held_mean=float(np.mean(ho)),
                                     alive=alive, seconds=round(time.time() - t_start, 1)))
            snapshot(P, step)
            print(f'[snap {step:5d}] train {np.mean(tr):.2f} dB  held-out {np.mean(ho):.2f} dB  alive {alive}  {time.time()-t_start:.0f}s', flush=True)
    if step == args.steps: break
    frac = step / args.steps
    k = TRAIN[rng.integers(NT)]
    bg = rng.random(3).astype(np.float32)
    target = RGB[k] + (1 - ALPHA[k])[..., None] * bg
    loss, img, g, _ = gs.forward_backward(P, cams[k], bg, l1(target))
    run_loss.append(loss)
    tt = step + 1
    for key, (lr0, lr1) in LR.items():
        lr = lr0 * (lr1 / lr0) ** frac
        m1[key] = B1 * m1[key] + (1 - B1) * g[key]; m2[key] = B2 * m2[key] + (1 - B2) * g[key] ** 2
        P[key] = (P[key] - lr * (m1[key] / (1 - B1 ** tt)) / (np.sqrt(m2[key] / (1 - B2 ** tt)) + EPS)).astype(np.float32)
    P['log_s'] = np.minimum(P['log_s'], LOG_S_MAX)
    if args.reset and step + 1 == args.reset:
        on = P['o'] > OFF + 1e-3
        P['o'][on] = np.minimum(P['o'][on], np.log(0.01 / 0.99)); m1['o'][:] = 0; m2['o'][:] = 0
        print(f'opacity reset at {step+1}', flush=True)
    if args.prune and (step + 1) >= PRUNE_FROM and (step + 1) <= PRUNE_UNTIL and (step + 1) % PRUNE_EVERY == 0:
        dead = gs.sigmoid(P['o']) < args.prune
        P['o'][dead] = OFF
        for kk in m1: m1[kk][dead] = 0; m2[kk][dead] = 0
        log.setdefault('pruned', []).append([step + 1, int((P['o'] <= OFF + 1e-3).sum())])
        print(f'prune at {step+1}: {int(dead.sum())} below {args.prune}, {log["pruned"][-1][1]} off in total', flush=True)
    if (step + 1) % 25 == 0:
        log['loss'].append([step + 1, float(np.mean(run_loss))]); run_loss = []
    if args.bench and step + 1 >= args.bench:
        print(f'bench: {(time.time()-t_last)/args.bench*1000:.0f} ms/step'); raise SystemExit
    if (step + 1) % 100 == 0:
        print(f'step {step+1:5d} L1 {log["loss"][-1][1]:.4f}  {(time.time()-t_last)/100*1000:.0f} ms/step', flush=True); t_last = time.time()
    if (step + 1) % CKPT == 0:
        log['seconds'] = round(time.time() - t_start, 1)
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        np.savez(ck + '.tmp.npz', step=step + 1, rng=np.array(rng.bit_generator.state, dtype=object),
                 **{'P_' + k: v for k, v in P.items()}, **{'m_' + k: v for k, v in m1.items()}, **{'v_' + k: v for k, v in m2.items()})
        os.replace(ck + '.tmp.npz', ck)
        json.dump(log, open(logp, 'w'))
log['seconds'] = round(time.time() - t_start, 1)
log['done'] = True
json.dump(log, open(logp, 'w'), indent=0)
print('done', log['seconds'], 's')
