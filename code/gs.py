"""A differentiable 3D Gaussian splatting rasterizer in plain NumPy, with hand-derived gradients.

Forward (as in Kerbl et al. 2023): project each 3D Gaussian to a 2D ellipse with the EWA / local-affine
approximation  Sigma2D = J W Sigma W^T J^T + 0.3 I,  sort by camera depth, and alpha-composite front to
back per pixel:  C = sum_i c_i a_i T_i + T_N bg,  T_i = prod_{j<i} (1 - a_j),  a_i = min(0.99, o_i G_i(p)).
Pixels are processed in 16x16 tiles; a Gaussian enters every tile its 3-sigma box touches; a < 1/255 is
treated as 0. Gradients flow back through compositing, the 2D conic, the projection, Sigma = R S S^T R^T,
the quaternion and the activations (exp scales, sigmoid colour and opacity). Checked in gradcheck.py.

Camera: OpenCV convention, x_cam = R x_world + t, pixel u = f x/z + W/2 (pixel i has its centre at i + 0.5).
"""
import numpy as np

TILE = 16
DILATE = 0.3
A_MIN, A_MAX = 1.0 / 255.0, 0.99


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def quat_to_rot(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rot_grad_to_quat(q, g):
    """dL/dq (unit quaternion, w x y z) from dL/dR, g: (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    G = lambda i, j: g[:, i, j]
    dw = 2 * (-z * G(0, 1) + y * G(0, 2) + z * G(1, 0) - x * G(1, 2) - y * G(2, 0) + x * G(2, 1))
    dx = 2 * (y * G(0, 1) + z * G(0, 2) + y * G(1, 0) - 2 * x * G(1, 1) - w * G(1, 2) + z * G(2, 0) + w * G(2, 1) - 2 * x * G(2, 2))
    dy = 2 * (-2 * y * G(0, 0) + x * G(0, 1) + w * G(0, 2) + x * G(1, 0) + z * G(1, 2) - w * G(2, 0) + z * G(2, 1) - 2 * y * G(2, 2))
    dz = 2 * (-2 * z * G(0, 0) - w * G(0, 1) + x * G(0, 2) + w * G(1, 0) - 2 * z * G(1, 1) + y * G(1, 2) + x * G(2, 0) + y * G(2, 1))
    return np.stack([dw, dx, dy, dz], 1)


def activate(P):
    qn = P['q'] / np.linalg.norm(P['q'], axis=1, keepdims=True)
    return dict(mu=P['mu'], s=np.exp(P['log_s']), qn=qn, c=sigmoid(P['c']), o=sigmoid(P['o']))


def project(A, R, t, f, W, H, cull=True):
    """Per-Gaussian screen-space quantities for one camera."""
    mu, s, qn = A['mu'], A['s'], A['qn']
    xc = mu @ R.T + t
    x, y, z = xc[:, 0], xc[:, 1], xc[:, 2]
    valid = z > 0.2
    z = np.where(valid, z, 1.0)
    Rq = quat_to_rot(qn)
    M = Rq * s[:, None, :]
    Sig = M @ M.transpose(0, 2, 1)
    N = len(mu)
    J = np.zeros((N, 2, 3), mu.dtype)
    J[:, 0, 0] = f / z; J[:, 0, 2] = -f * x / z ** 2
    J[:, 1, 1] = f / z; J[:, 1, 2] = -f * y / z ** 2
    T = J @ R
    S2 = T @ Sig @ T.transpose(0, 2, 1)
    Av, Bv, Cv = S2[:, 0, 0] + DILATE, S2[:, 0, 1], S2[:, 1, 1] + DILATE
    det = Av * Cv - Bv * Bv
    ca, cb, cc = Cv / det, -Bv / det, Av / det
    mx, my = f * x / z + W / 2, f * y / z + H / 2
    if cull:
        mid = 0.5 * (Av + Cv)
        lam = mid + np.sqrt(np.maximum(0.1, mid * mid - det))
        rad = np.ceil(3.0 * np.sqrt(lam))
    else:
        rad = np.full(N, 1e9)
    valid &= det > 0
    return dict(xc=xc, x=x, y=y, z=z, valid=valid, Rq=Rq, M=M, Sig=Sig, J=J, T=T, Av=Av, Bv=Bv, Cv=Cv, det=det,
                ca=ca, cb=cb, cc=cc, mx=mx, my=my, rad=rad)


def rasterize(A, pr, W, H, bg, keep=False, cull=True):
    """Returns image (H,W,3), final transmittance (H,W) and, if keep, the per-tile cache for backward."""
    order = np.argsort(pr['z'], kind='stable')
    order = order[pr['valid'][order]]
    mx, my, rad = pr['mx'][order], pr['my'][order], pr['rad'][order]
    img = np.zeros((H, W, 3), A['mu'].dtype); Tfin = np.ones((H, W), A['mu'].dtype)
    cache = []
    a_min = A_MIN if cull else 0.0
    for ty in range(0, H, TILE):
        for tx in range(0, W, TILE):
            x1, y1 = min(tx + TILE, W), min(ty + TILE, H)
            sel = (mx + rad >= tx) & (mx - rad < x1) & (my + rad >= ty) & (my - rad < y1)
            idx = order[sel]
            py, px = np.mgrid[ty:y1, tx:x1]
            px = px.ravel() + 0.5; py = py.ravel() + 0.5
            if len(idx) == 0:
                img[ty:y1, tx:x1] = bg
                if keep: cache.append(None)
                continue
            dx = px[None, :] - pr['mx'][idx, None]; dy = py[None, :] - pr['my'][idx, None]
            ca, cb, cc = pr['ca'][idx, None], pr['cb'][idx, None], pr['cc'][idx, None]
            power = np.minimum(-0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy, 0.0)
            G = np.exp(power)
            a = A['o'][idx, None] * G
            m = (a >= a_min) & (a <= A_MAX)
            a = np.where(a < a_min, 0.0, np.minimum(a, A_MAX))
            la = np.log1p(-a)
            cum = np.cumsum(la, 0)
            Tinc = np.exp(cum)
            Texc = np.exp(cum - la)
            w = a * Texc
            Tf = Tinc[-1]
            col = w.T @ A['c'][idx] + Tf[:, None] * bg[None, :]
            img[ty:y1, tx:x1] = col.reshape(y1 - ty, x1 - tx, 3)
            Tfin[ty:y1, tx:x1] = Tf.reshape(y1 - ty, x1 - tx)
            if keep:
                cache.append(dict(idx=idx, ys=slice(ty, y1), xs=slice(tx, x1), dx=dx, dy=dy, G=G, a=a, m=m, Texc=Texc, w=w, Tf=Tf))
    return img, Tfin, cache


def render(P, cam, bg=None, cull=True):
    R, t, f, W, H = cam
    A = activate(P)
    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    pr = project(A, R, t, f, W, H, cull)
    img, Tf, _ = rasterize(A, pr, W, H, bg, cull=cull)
    return img, 1 - Tf


def forward_backward(P, cam, bg, loss_grad, cull=True):
    """loss_grad(img) -> (loss, dL/dimg). Returns loss, img, grads dict matching P, and per-Gaussian |dL/dmean2D|."""
    R, t, f, W, H = cam
    A = activate(P)
    pr = project(A, R, t, f, W, H, cull)
    img, Tf, cache = rasterize(A, pr, W, H, bg, keep=True, cull=cull)
    loss, gimg = loss_grad(img)
    N = len(P['mu']); dt = P['mu'].dtype
    dc = np.zeros((N, 3), dt); do = np.zeros(N, dt)
    dA = np.zeros(N, dt); dB = np.zeros(N, dt); dC = np.zeros(N, dt)
    dmx = np.zeros(N, dt); dmy = np.zeros(N, dt)
    for ch in cache:
        if ch is None: continue
        idx = ch['idx']
        g = gimg[ch['ys'], ch['xs']].reshape(-1, 3)            # (P,3)
        w, a, Texc, G = ch['w'], ch['a'], ch['Texc'], ch['G']
        dc[idx] += w @ g
        sg = A['c'][idx] @ g.T                                  # c_i . g_p   (K,P)
        ws = w * sg
        S = ws.sum(0)[None, :] - np.cumsum(ws, 0) + (ch['Tf'] * (g @ bg))[None, :]
        da = (Texc * sg - S / (1 - a)) * ch['m']
        do[idx] += (da * G).sum(1)
        dp = da * A['o'][idx, None] * G
        dx, dy = ch['dx'], ch['dy']
        dA[idx] += (dp * (-0.5 * dx * dx)).sum(1)
        dB[idx] += (dp * (-dx * dy)).sum(1)
        dC[idx] += (dp * (-0.5 * dy * dy)).sum(1)
        ca, cb, cc = pr['ca'][idx, None], pr['cb'][idx, None], pr['cc'][idx, None]
        dmx[idx] += (dp * (ca * dx + cb * dy)).sum(1)
        dmy[idx] += (dp * (cb * dx + cc * dy)).sum(1)
    # conic (ca,cb,cc) <- 2D covariance (Av,Bv,Cv)
    Av, Bv, Cv, det = pr['Av'], pr['Bv'], pr['Cv'], pr['det']
    d2 = det * det
    gAv = (-Cv * Cv * dA + Bv * Cv * dB - Bv * Bv * dC) / d2
    gBv = (2 * Bv * Cv * dA - (det + 2 * Bv * Bv) * dB + 2 * Av * Bv * dC) / d2
    gCv = (-Bv * Bv * dA + Av * Bv * dB - Av * Av * dC) / d2
    Gs = np.zeros((N, 2, 2), dt)
    Gs[:, 0, 0] = gAv; Gs[:, 1, 1] = gCv; Gs[:, 0, 1] = Gs[:, 1, 0] = 0.5 * gBv
    T, Sig, M, Rq, J = pr['T'], pr['Sig'], pr['M'], pr['Rq'], pr['J']
    GSig = T.transpose(0, 2, 1) @ Gs @ T                        # dL/dSigma (3x3, symmetric)
    GT = 2 * Gs @ T @ Sig                                       # dL/dT
    GJ = GT @ R.T                                               # T = J R
    x, y, z = pr['x'], pr['y'], pr['z']
    gx = GJ[:, 0, 2] * (-f / z ** 2) + dmx * (f / z)
    gy = GJ[:, 1, 2] * (-f / z ** 2) + dmy * (f / z)
    gz = (GJ[:, 0, 0] * (-f / z ** 2) + GJ[:, 0, 2] * (2 * f * x / z ** 3) + GJ[:, 1, 1] * (-f / z ** 2)
          + GJ[:, 1, 2] * (2 * f * y / z ** 3) + dmx * (-f * x / z ** 2) + dmy * (-f * y / z ** 2))
    gxc = np.stack([gx, gy, gz], 1) * pr['valid'][:, None]
    dmu = gxc @ R
    GM = 2 * GSig @ M
    ds = (GM * Rq).sum(1)                                       # M_ij = Rq_ij s_j
    GR = GM * A['s'][:, None, :]
    dqn = rot_grad_to_quat(A['qn'], GR)
    qraw = P['q']; nq = np.linalg.norm(qraw, axis=1, keepdims=True); qn = A['qn']
    dq = (dqn - qn * (dqn * qn).sum(1, keepdims=True)) / nq
    V = pr['valid'][:, None]
    grads = dict(mu=dmu, log_s=ds * A['s'] * V, q=dq * V, c=dc * A['c'] * (1 - A['c']), o=do * A['o'] * (1 - A['o']))
    g2d = np.hypot(dmx, dmy)   # not used for densification here; kept for diagnostics
    return loss, img, grads, g2d
