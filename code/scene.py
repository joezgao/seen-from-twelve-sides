"""The 'photographs': a procedural river stone, sphere-traced in NumPy from 16 known cameras.

12 cameras are the training set (three rings of four), 4 more sit between them and are never trained on.
Output: data/gt.npz with premultiplied RGB, alpha (coverage from 3x3 supersampling) and the cameras.
Lighting is baked into the stone's colour (fixed lights in world space), so what each camera sees is
view-independent apart from occlusion, which is what colour-only Gaussians can represent.
"""
import numpy as np, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 96          # image size
SS = 3              # supersampling per axis
FOV = 38.0          # degrees, vertical = horizontal
DIST = 3.15         # camera distance from the stone's centre

# ---------------------------------------------------------------- noise
rng = np.random.default_rng(7)
LAT = rng.random((32, 32, 32)).astype(np.float64)

def vnoise(p):
    """trilinear value noise with smoothstep, p: (...,3) -> (...) in [0,1]"""
    q = np.floor(p); f = p - q; f = f * f * (3 - 2 * f)
    i = q.astype(np.int64) % 32; j = (i + 1) % 32
    x0, y0, z0 = i[..., 0], i[..., 1], i[..., 2]; x1, y1, z1 = j[..., 0], j[..., 1], j[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    c00 = LAT[x0, y0, z0] * (1 - fx) + LAT[x1, y0, z0] * fx
    c10 = LAT[x0, y1, z0] * (1 - fx) + LAT[x1, y1, z0] * fx
    c01 = LAT[x0, y0, z1] * (1 - fx) + LAT[x1, y0, z1] * fx
    c11 = LAT[x0, y1, z1] * (1 - fx) + LAT[x1, y1, z1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz

def fbm(p, oct=3):
    s, a, tot = 0.0, 0.5, 0.0
    for k in range(oct):
        s = s + a * vnoise(p * (2.0 ** k) + 17.3 * k); tot += a; a *= 0.5
    return s / tot

# ---------------------------------------------------------------- the stone
RADII = np.array([1.05, 0.60, 0.80])
th = np.radians(18.0)   # a small tilt so it is not axis-aligned
TILT = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
       np.array([[1, 0, 0], [0, np.cos(0.25), -np.sin(0.25)], [0, np.sin(0.25), np.cos(0.25)]])

def local(p): return p @ TILT          # world -> stone frame (TILT is orthonormal)

def sdf(p):
    q = local(p)
    base = (np.linalg.norm(q / RADII, axis=-1) - 1.0) * RADII.min()
    disp = 0.10 * (fbm(q * 1.1 + 3.0, 2) - 0.5) + 0.025 * (fbm(q * 4.0, 2) - 0.5)
    return base - disp

def normal(p, e=2e-3):
    n = np.stack([sdf(p + np.array([e, 0, 0])) - sdf(p - np.array([e, 0, 0])),
                  sdf(p + np.array([0, e, 0])) - sdf(p - np.array([0, e, 0])),
                  sdf(p + np.array([0, 0, e])) - sdf(p - np.array([0, 0, e]))], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)

def albedo(p):
    q = local(p)
    slate = np.array([0.30, 0.33, 0.33]); warm = np.array([0.52, 0.47, 0.41])
    m = np.clip((fbm(q * 1.8 + 5.0, 3) - 0.35) * 2.2, 0, 1)[..., None]
    col = slate * (1 - m) + warm * m
    sp = fbm(q * 9.0, 2)                                  # speckles
    col = col * (0.82 + 0.36 * np.clip((sp - 0.45) * 3, 0, 1)[..., None])
    nb = np.array([0.35, 1.0, 0.15]); nb /= np.linalg.norm(nb)   # quartz band (in stone frame)
    d = q @ nb + 0.08 * (fbm(q * 2.5 + 9.0, 2) - 0.5) - 0.06
    band = np.clip(1 - (np.abs(d) - 0.07) / 0.025, 0, 1)[..., None]
    d2 = q @ np.array([0.9, -0.3, 0.3]) / np.linalg.norm([0.9, -0.3, 0.3]) - 0.42   # thin second vein
    band2 = np.clip(1 - (np.abs(d2) - 0.012) / 0.012, 0, 1)[..., None] * 0.85
    cream = np.array([0.90, 0.86, 0.78])
    col = col * (1 - band) + cream * band
    col = col * (1 - band2) + cream * band2
    return col

L1 = np.array([-0.55, 0.75, 0.45]); L1 /= np.linalg.norm(L1)
L2 = np.array([0.6, 0.15, -0.75]); L2 /= np.linalg.norm(L2)

def shade(p):
    n = normal(p)
    ao = np.ones(p.shape[:-1])                            # 4-tap sdf ambient occlusion
    for k, h in enumerate([0.04, 0.08, 0.14, 0.22]):
        ao -= (h - sdf(p + n * h)) * (0.5 ** k) * 1.6
    ao = np.clip(ao, 0.35, 1.0)
    diff = 1.05 * np.clip(n @ L1, 0, None) + 0.28 * np.clip(n @ L2, 0, None)
    amb = 0.22 + 0.10 * (0.5 + 0.5 * n[..., 1]) + 0.14 * np.clip(-n[..., 1], 0, None)
    light = (amb * ao)[..., None] + diff[..., None] * np.array([1.0, 0.97, 0.92]) * (0.6 + 0.4 * ao[..., None])
    return np.clip(albedo(p) * light, 0, 1)

# ---------------------------------------------------------------- cameras (OpenCV: x right, y down, z forward)
def look_at(pos):
    f = -pos / np.linalg.norm(pos)
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f]); t = -R @ pos
    return R, t

def cam_pos(az, el):
    az, el = np.radians(az), np.radians(el)
    return DIST * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])

TRAIN = [(az + off, el) for el, off in [(52, 0), (14, 45), (-24, 0)] for az in (0, 90, 180, 270)]
HELD = [(22.5, 32), (112.5, -6), (202.5, 32), (292.5, -6)]
FOCAL = 0.5 * W / np.tan(np.radians(FOV) / 2)

def render(R, t):
    n = W * SS
    s = (np.arange(n) + 0.5) / SS
    u, v = np.meshgrid(s, s)
    dc = np.stack([(u - W / 2) / FOCAL, (v - H / 2) / FOCAL, np.ones_like(u)], -1).reshape(-1, 3)
    d = dc @ R; d /= np.linalg.norm(d, axis=-1, keepdims=True)
    o = -R.T @ t
    b = d @ o; c = o @ o - 1.35 ** 2; disc = b * b - c       # bounding sphere
    hit = np.zeros(len(d), bool); tt = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), 1e9)
    act = disc > 0
    for _ in range(90):
        idx = np.nonzero(act)[0]
        if not len(idx): break
        p = o + d[idx] * tt[idx, None]
        dist = sdf(p)
        tt[idx] += dist * 0.7
        done = dist < 4e-4
        hit[idx[done]] = True
        far = tt[idx] > DIST + 1.5
        act[idx[done | far]] = False
    col = np.zeros((len(d), 3))
    hi = np.nonzero(hit)[0]
    col[hi] = shade(o + d[hi] * tt[hi, None])
    col = col.reshape(H, SS, W, SS, 3).mean((1, 3))
    alpha = hit.astype(float).reshape(H, SS, W, SS).mean((1, 3))
    return col, alpha     # col is premultiplied (black where missed)

if __name__ == '__main__':
    t0 = time.time()
    cams = TRAIN + HELD
    Rs, ts, imgs, alphas = [], [], [], []
    for k, (az, el) in enumerate(cams):
        R, t = look_at(cam_pos(az, el)); c, a = render(R, t)
        Rs.append(R); ts.append(t); imgs.append(c); alphas.append(a)
        print(f'cam {k:2d} az {az:6.1f} el {el:4d}  coverage {a.mean():.3f}  {time.time()-t0:.1f}s', flush=True)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    np.savez(os.path.join(HERE, 'data', 'gt.npz'), R=np.array(Rs), t=np.array(ts), rgb=np.array(imgs, np.float32),
             alpha=np.array(alphas, np.float32), focal=FOCAL, W=W, H=H, n_train=len(TRAIN),
             az_el=np.array(cams, float))
    from PIL import Image
    row = lambda ims: np.concatenate(ims, 1)
    grid = np.concatenate([row(imgs[0:4]), row(imgs[4:8]), row(imgs[8:12]), row(imgs[12:16])], 0)
    Image.fromarray((np.clip(grid, 0, 1) * 255).astype(np.uint8)).resize((grid.shape[1] * 2, grid.shape[0] * 2), Image.NEAREST).save(os.path.join(HERE, 'data', 'gt_grid.png'))
    print('done', time.time() - t0)
