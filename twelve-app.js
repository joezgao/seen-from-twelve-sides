
(function(){
'use strict';
const $ = (id) => document.getElementById(id);
const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const fmt = (x, d=1) => x.toFixed(d);
const int = (x) => Math.round(x).toLocaleString('en-US');
const N = META.n, S = META.steps.length, F = 14;
const last = S - 1;

// ------------------------------------------------------------ decode float16 snapshots
const raw = atob(BLOB); const u8 = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);
const h16 = new Uint16Array(u8.buffer);
const H2F = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  H2F[h] = e === 0 ? s * m * Math.pow(2, -24) : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * Math.pow(2, e - 15);
}
const SNAP = [];
for (let k = 0; k < S; k++) { const a = new Float32Array(N * F); const o = k * N * F; for (let i = 0; i < N * F; i++) a[i] = H2F[h16[o + i]]; SNAP.push(a); }

// ------------------------------------------------------------ copy from the training log
const lastS = META.snaps[last];
$('lede').innerHTML = `Twelve rendered photographs of a stone, and <b>${int(N)}</b> Gaussians fitted to them by gradient descent, written from scratch in NumPy. Scrub the training to watch the cloud become the stone.`;
// the input, shown first: atlas slot k is camera k; rows 1 to 3 are the three training rings, row 4 the held-back cameras
const NH = META.held_ids.length;
$('photos').style.setProperty('--atlas', `url(${ATLAS})`);
$('photoGrid').setAttribute('aria-label', `The ${META.n_train} training photographs, ${META.img} by ${META.img} px, and ${NH} empty slots for the held-back cameras`);
$('photoCap').innerHTML = `<span class="j">The ${META.n_train} photographs it learns from</span>, ${META.img}&nbsp;px each<span class="capKept">, one row per ring of cameras. Dashed: the ${NH} held back to test it.</span><span class="capKeptPh">; ${NH} more are held back to test it.</span>`;
const phSlots = META.cams.map((c, k) => {
  const el = document.createElement('div'); el.className = 'ph' + (k >= META.n_train ? ' kept' : '');
  el.style.backgroundPosition = `${(k % 4) / 3 * 100}% ${Math.floor(k / 4) / 3 * 100}%`;
  el.style.transitionDelay = `${Math.floor(k / 4) * 140 + (k % 4) * 35}ms`;
  $('photoGrid').appendChild(el); return el;
});
requestAnimationFrame(() => requestAnimationFrame(() => $('photos').classList.add('in')));

// ------------------------------------------------------------ three.js setup
let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas: $('view'), antialias: false, alpha: false, preserveDrawingBuffer: false }); }
catch (e) { $('fallback').style.display = 'grid'; return; }
renderer.setClearColor(0x000000, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(META.fov, 1, 0.05, 100);

const quad = new THREE.InstancedBufferGeometry();
quad.setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1,-1,0, 1,-1,0, 1,1,0, -1,1,0]), 3));
quad.setIndex([0,1,2, 0,2,3]);
const aCenter = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
const aCovA = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
const aCovB = new THREE.InstancedBufferAttribute(new Float32Array(N * 3), 3);
const aColor = new THREE.InstancedBufferAttribute(new Float32Array(N * 4), 4);
[aCenter, aCovA, aCovB, aColor].forEach((a) => a.setUsage(THREE.DynamicDrawUsage));
quad.setAttribute('iCenter', aCenter); quad.setAttribute('iCovA', aCovA); quad.setAttribute('iCovB', aCovB); quad.setAttribute('iColor', aColor);
quad.instanceCount = N;
quad.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 10);

const splatMat = new THREE.ShaderMaterial({
  uniforms: { uViewport: { value: new THREE.Vector2(1, 1) }, uDilWorld: { value: 0 } },
  vertexShader: `
    attribute vec3 iCenter; attribute vec3 iCovA; attribute vec3 iCovB; attribute vec4 iColor;
    uniform vec2 uViewport; uniform float uDilWorld;
    varying vec4 vColor; varying vec2 vPos;
    void main(){
      vec4 cam = modelViewMatrix * vec4(iCenter, 1.0);
      float z = -cam.z;
      if (z < 0.1) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
      vec4 clip = projectionMatrix * cam;
      float fx = projectionMatrix[0][0] * 0.5 * uViewport.x;
      float fy = projectionMatrix[1][1] * 0.5 * uViewport.y;
      mat3 S = mat3(iCovA.x, iCovA.y, iCovA.z,  iCovA.y, iCovB.x, iCovB.y,  iCovA.z, iCovB.y, iCovB.z);
      mat3 J = mat3(fx / z, 0.0, 0.0,  0.0, fy / z, 0.0,  fx * cam.x / (z * z), fy * cam.y / (z * z), 0.0);
      mat3 T = J * mat3(modelViewMatrix);
      mat3 C = T * S * transpose(T);
      // the same 0.3 px low-pass the fit used, measured in training pixels at the training distance
      float dil = max(0.3, uDilWorld * (fx / z) * (fx / z));
      float a = C[0][0] + dil, b = C[0][1], c = C[1][1] + dil;
      float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
      float l1 = mid + rad, l2 = max(mid - rad, 0.05);
      vec2 v1 = abs(b) < 1e-7 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(vec2(b, l1 - a));
      vec2 v2 = vec2(-v1.y, v1.x);
      vec2 off = position.x * min(3.0 * sqrt(l1), 1024.0) * v1 + position.y * min(3.0 * sqrt(l2), 1024.0) * v2;
      vPos = position.xy * 3.0;
      vColor = iColor;
      gl_Position = clip + vec4(off / (0.5 * uViewport) * clip.w, 0.0, 0.0);
    }`,
  fragmentShader: `
    precision highp float;
    varying vec4 vColor; varying vec2 vPos;
    void main(){
      float a = min(0.99, vColor.a * exp(-0.5 * dot(vPos, vPos)));
      if (a < 1.0 / 255.0) discard;
      gl_FragColor = vec4(vColor.rgb * a, a);
    }`,
  transparent: true, depthTest: false, depthWrite: false,
  blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
});
const splats = new THREE.Mesh(quad, splatMat);
splats.frustumCulled = false; splats.renderOrder = 0;
scene.add(splats);
let DIL_WORLD = 0.3 * Math.pow(META.dist / META.focal, 2);

// ------------------------------------------------------------ current Gaussians (eased between logged snapshots)
const cur = { mu: new Float32Array(N * 3), cov: new Float32Array(N * 6), col: new Float32Array(N * 4) };
let shownT = last, dirtyParams = true, dirtySort = true;
function buildParams(t) {
  const k0 = Math.min(Math.floor(t), last), k1 = Math.min(k0 + 1, last), u = t - k0;
  const A = SNAP[k0], B = SNAP[k1];
  for (let i = 0; i < N; i++) {
    const o = i * F;
    const L = (j) => A[o + j] + (B[o + j] - A[o + j]) * u;
    cur.mu[i * 3] = L(0); cur.mu[i * 3 + 1] = L(1); cur.mu[i * 3 + 2] = L(2);
    const sx = Math.exp(L(3)), sy = Math.exp(L(4)), sz = Math.exp(L(5));
    let w = L(6), x = L(7), y = L(8), z = L(9);
    if (u > 0 && A[o + 6] * B[o + 6] + A[o + 7] * B[o + 7] + A[o + 8] * B[o + 8] + A[o + 9] * B[o + 9] < 0) {
      w = A[o + 6] - (B[o + 6] + A[o + 6]) * u; x = A[o + 7] - (B[o + 7] + A[o + 7]) * u; y = A[o + 8] - (B[o + 8] + A[o + 8]) * u; z = A[o + 9] - (B[o + 9] + A[o + 9]) * u;
    }
    const n = Math.hypot(w, x, y, z) || 1; w /= n; x /= n; y /= n; z /= n;
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
    const a = sx * sx, b = sy * sy, c = sz * sz, q = i * 6;
    cur.cov[q] = r00 * r00 * a + r01 * r01 * b + r02 * r02 * c;
    cur.cov[q + 1] = r00 * r10 * a + r01 * r11 * b + r02 * r12 * c;
    cur.cov[q + 2] = r00 * r20 * a + r01 * r21 * b + r02 * r22 * c;
    cur.cov[q + 3] = r10 * r10 * a + r11 * r11 * b + r12 * r12 * c;
    cur.cov[q + 4] = r10 * r20 * a + r11 * r21 * b + r12 * r22 * c;
    cur.cov[q + 5] = r20 * r20 * a + r21 * r21 * b + r22 * r22 * c;
    cur.col[i * 4] = L(10); cur.col[i * 4 + 1] = L(11); cur.col[i * 4 + 2] = L(12); cur.col[i * 4 + 3] = L(13);
  }
}
const depth = new Float32Array(N), order = new Uint32Array(N);
const keys = new Float64Array(N);
function fillSorted(cam) {
  cam.updateMatrixWorld(); const m = cam.matrixWorldInverse.elements;
  for (let i = 0; i < N; i++) {
    const d = m[2] * cur.mu[i * 3] + m[6] * cur.mu[i * 3 + 1] + m[10] * cur.mu[i * 3 + 2] + m[14];
    keys[i] = d * 1e6 + i * 1e-3; // back to front: most negative view-z first
  }
  const idx = Array.from({ length: N }, (_, i) => i).sort((a, b) => keys[a] - keys[b]);
  const C = aCenter.array, A = aCovA.array, B = aCovB.array, K = aColor.array;
  for (let j = 0; j < N; j++) {
    const i = idx[j];
    C[j * 3] = cur.mu[i * 3]; C[j * 3 + 1] = cur.mu[i * 3 + 1]; C[j * 3 + 2] = cur.mu[i * 3 + 2];
    A[j * 3] = cur.cov[i * 6]; A[j * 3 + 1] = cur.cov[i * 6 + 1]; A[j * 3 + 2] = cur.cov[i * 6 + 2];
    B[j * 3] = cur.cov[i * 6 + 3]; B[j * 3 + 1] = cur.cov[i * 6 + 4]; B[j * 3 + 2] = cur.cov[i * 6 + 5];
    K[j * 4] = cur.col[i * 4]; K[j * 4 + 1] = cur.col[i * 4 + 1]; K[j * 4 + 2] = cur.col[i * 4 + 2]; K[j * 4 + 3] = cur.col[i * 4 + 3];
  }
  aCenter.needsUpdate = aCovA.needsUpdate = aCovB.needsUpdate = aColor.needsUpdate = true;
}

// ------------------------------------------------------------ the sixteen cameras
const atlasImg = new Image(); atlasImg.src = ATLAS;
const atlasTex = new THREE.Texture(atlasImg); atlasImg.onload = () => { atlasTex.needsUpdate = true; requestDraw(); };
const ACC = new THREE.Color(0x9fd8b4), INK = new THREE.Color(0xecebe5);
const camGroups = [];
function camBasis(c) { // OpenCV rows r (right), d (down), f (forward); centre = -R^T t
  const R = c.R, t = c.t;
  const r = new THREE.Vector3(R[0], R[1], R[2]), d = new THREE.Vector3(R[3], R[4], R[5]), f = new THREE.Vector3(R[6], R[7], R[8]);
  const C = new THREE.Vector3(-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]));
  return { r, d, f, C };
}
META.cams.forEach((c, k) => {
  const { r, d, f, C } = camBasis(c), held = k >= META.n_train;
  const depthP = 0.5, half = depthP * (META.img / 2) / META.focal;
  const corner = (sx, sy) => C.clone().addScaledVector(f, depthP).addScaledVector(r, sx * half).addScaledVector(d, sy * half);
  const tl = corner(-1, -1), tr = corner(1, -1), br = corner(1, 1), bl = corner(-1, 1);
  const pts = [C, tl, C, tr, C, br, C, bl, tl, tr, tr, br, br, bl, bl, tl];
  const g = new THREE.Group();
  const lm = new THREE.LineBasicMaterial({ color: held ? INK : ACC, transparent: true, opacity: held ? 0.45 : 0.8, depthWrite: false });
  g.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), lm));
  if (!held) {
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array([tl.x, tl.y, tl.z, tr.x, tr.y, tr.z, br.x, br.y, br.z, bl.x, bl.y, bl.z]), 3));
    const col = k % 4, row = Math.floor(k / 4), u0 = col / 4, u1 = (col + 1) / 4, v1 = 1 - row / 4, v0 = 1 - (row + 1) / 4;
    pg.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([u0, v1, u1, v1, u1, v0, u0, v0]), 2));
    pg.setIndex([0, 2, 1, 0, 3, 2]);
    g.add(new THREE.Mesh(pg, new THREE.MeshBasicMaterial({ map: atlasTex, side: THREE.DoubleSide, transparent: true, opacity: 0.92, depthWrite: false })));
  }
  g.userData.C = C; g.visible = false;
  scene.add(g); camGroups.push(g);
});
atlasTex.magFilter = THREE.LinearFilter; atlasTex.minFilter = THREE.LinearFilter; atlasTex.generateMipmaps = false;

// ------------------------------------------------------------ orbit
const view = { az: 0.6, el: 0.32, dist: 0, target: null };
let baseDist = 4.6;
// frame the stone near the size it was fitted at (96 px photographs): about 3x on a laptop screen, not full-bleed
const SIDE = matchMedia('(max-height:500px) and (orientation:landscape)');   // a phone on its side
const FIT = { desk: 2.8, phone: 1.85, camsDesk: 1.95, camsPhone: 1.55, band: 1.2 };
function fitDist(cams) { const a = innerWidth / innerHeight, v = THREE.MathUtils.degToRad(META.fov) / 2, h = Math.atan(Math.tan(v) * a), ph = innerWidth < 721; return (cams ? (ph ? FIT.camsPhone : FIT.camsDesk) : (ph ? FIT.phone : FIT.desk)) / Math.tan(Math.min(v, h)); }
function placeCamera(cam, az, el, dist) {
  cam.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az));
  cam.up.set(0, 1, 0); cam.lookAt(0, 0, 0);
}
let anim = null; // camera glide
function glideTo(az, el, dist, ms = 700) {
  if (reduced) { view.az = az; view.el = el; view.dist = dist; dirtySort = true; requestDraw(); return; }
  let d = az - view.az; d = ((d + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  anim = { t0: performance.now(), ms, from: { az: view.az, el: view.el, dist: view.dist }, to: { az: view.az + d, el, dist } };
  requestDraw();
}
let userTouched = false, autoSpin = !reduced;
const cv = $('view'); let drag = null; const pointers = new Map();
cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY }); drag = { x: e.clientX, y: e.clientY }; stopSpin(); });
cv.addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()]; const dNow = Math.hypot(a.x - b.x, a.y - b.y);
    const others = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
    const dPrev = Math.hypot(prev.x - others.x, prev.y - others.y);
    if (dPrev > 0) zoom(dPrev / dNow);
    return;
  }
  view.az -= (e.clientX - prev.x) * 0.008; view.el = clampEl(view.el + (e.clientY - prev.y) * 0.006);
  anim = null; dirtySort = true; requestDraw();
});
const up = (e) => { pointers.delete(e.pointerId); if (!pointers.size) drag = null; };
cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
cv.addEventListener('wheel', (e) => { e.preventDefault(); stopSpin(); zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
cv.addEventListener('keydown', (e) => {
  const k = e.key; let used = true;
  if (k === 'ArrowLeft') view.az += 0.12; else if (k === 'ArrowRight') view.az -= 0.12;
  else if (k === 'ArrowUp') view.el = clampEl(view.el + 0.1); else if (k === 'ArrowDown') view.el = clampEl(view.el - 0.1);
  else if (k === '+' || k === '=') zoom(0.9); else if (k === '-' || k === '_') zoom(1.1);
  else used = false;
  if (used) { e.preventDefault(); stopSpin(); anim = null; dirtySort = true; requestDraw(); }
});
function clampEl(x) { return Math.max(-1.35, Math.min(1.35, x)); }
function zoom(f) { view.dist = Math.max(1.8, Math.min(18, view.dist * f)); anim = null; dirtySort = true; requestDraw(); }
function stopSpin() { userTouched = true; autoSpin = false; }

// ------------------------------------------------------------ training scrubber, curve, readout
const slider = $('step'); slider.max = last;
const pmin = Math.floor(Math.min(...META.snaps.map((s) => Math.min(s.train_mean, s.held_mean)))), pmax = Math.ceil(Math.max(...META.snaps.map((s) => s.train_mean)));
$('ytick').textContent = `PSNR ${pmin} to ${pmax} dB`;
function drawCurve(t) {
  const W = 520, H = 58, x = (i) => (i / last) * W, y = (v) => H - 4 - (v - pmin) / (pmax - pmin) * (H - 10);
  const line = (key) => META.snaps.map((s, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(s[key]).toFixed(1)}`).join('');
  $('curve').innerHTML = `<line class="grid" x1="0" x2="${W}" y1="${H - 0.5}" y2="${H - 0.5}"/>` +
    `<path class="tr" d="${line('train_mean')}" vector-effect="non-scaling-stroke"/><path class="ho" d="${line('held_mean')}" vector-effect="non-scaling-stroke"/>` +
    `<line class="now" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${H}" vector-effect="non-scaling-stroke"/>`;
}
function readout(k) {
  const s = META.snaps[k];
  $('readout').innerHTML = `<span class="step">Step ${int(s.step)}</span>` +
    `PSNR <b>${fmt(s.train_mean)} dB</b> on the training views, <b>${fmt(s.held_mean)} dB</b> on the held-out views.<br>${int(s.alive)} of ${int(N)} Gaussians have opacity above 0.05.`;
  slider.setAttribute('aria-valuetext', `Step ${s.step}: training ${fmt(s.train_mean)} dB, held-out ${fmt(s.held_mean)} dB`);
  slider.style.setProperty('--fill', (k / last * 100) + '%');
  drawCurve(k);
  if (held.on) updateCompare();
}
let tween = null, playing = null;
function goTo(k, ms) {
  k = Math.max(0, Math.min(last, k)); slider.value = k; readout(k);
  if (reduced || !ms) { shownT = k; tween = null; dirtyParams = true; requestDraw(); return; }
  tween = { t0: performance.now(), from: shownT, to: k, ms }; requestDraw();
}
slider.addEventListener('input', () => { stopPlay(); goTo(+slider.value, 450); });
function setPlayIcon(on) { $('playIcon').innerHTML = on ? '<path d="M3 1.5h3v11H3zM8 1.5h3v11H8z"/>' : '<path d="M3 1.5v11l9-5.5z"/>'; $('play').setAttribute('aria-label', on ? 'Pause training' : 'Play training'); }
function stopPlay() { if (playing) { clearTimeout(playing); playing = null; setPlayIcon(false); } }
function play() {
  if (+slider.value >= last) goTo(0, 0);
  setPlayIcon(true);
  const tick = () => { const k = +slider.value + 1; if (k > last) { stopPlay(); return; } goTo(k, reduced ? 0 : 520); playing = setTimeout(tick, reduced ? 700 : 640); };
  playing = setTimeout(tick, 350);
}
$('play').addEventListener('click', () => (playing ? stopPlay() : play()));

// ------------------------------------------------------------ toggles
const camsBtn = $('camsBtn'); let camsOn = false;
camsBtn.addEventListener('click', () => {
  camsOn = !camsOn; camsBtn.setAttribute('aria-pressed', camsOn);
  camGroups.forEach((g) => (g.visible = camsOn)); $('camkey').classList.toggle('on', camsOn);
  glideTo(view.az, camsOn ? Math.max(view.el, 0.5) : view.el, camsOn ? Math.max(fitDist(true) * 2.1, 11) : baseDist, 900);
});
const howEl = $('how'), howBtn = $('howBtn');
function setHow(on) {
  howEl.classList.toggle('on', on); howBtn.setAttribute('aria-expanded', on); howBtn.setAttribute('aria-pressed', on);
  if (on) $('howClose').focus(); else howBtn.focus();
}
howBtn.addEventListener('click', () => setHow(!howEl.classList.contains('on')));
$('howClose').addEventListener('click', () => setHow(false));
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && howEl.classList.contains('on')) setHow(false); });

// held-out comparison: hold H (or press the button) to fly to a held-out camera and see its photograph
const held = { on: false, j: -1, saved: null };
const mini = $('mini'), mctx = mini.getContext('2d');
const rt = new THREE.WebGLRenderTarget(META.img, META.img);
const miniCam = new THREE.PerspectiveCamera(META.fov, 1, 0.05, 100);
const pix = new Uint8Array(META.img * META.img * 4), imgData = mctx.createImageData(META.img, META.img);
const gctx = $('gt').getContext('2d'), dctx = $('diff').getContext('2d'), diffData = dctx.createImageData(META.img, META.img);
let gtData = null, gtId = -1;
function gtPixels(id) { // the held-out photograph, copied out of the atlas at its own 96 px
  if (!atlasImg.complete || !atlasImg.naturalWidth) { gtData = null; return; }
  if (gtId === id && gtData) return;
  const n = META.img; gctx.clearRect(0, 0, n, n); gctx.drawImage(atlasImg, (id % 4) * n, Math.floor(id / 4) * n, n, n, 0, 0, n, n);
  gtData = gctx.getImageData(0, 0, n, n); gtId = id;
}
function updateCompare() {
  const j = held.j, id = META.held_ids[j], c = META.cams[id], s = META.snaps[+slider.value];
  gtPixels(id);
  $('cmpText').innerHTML = `Held-out camera ${j + 1} of ${META.held_ids.length}. At step ${int(s.step)} the fit scores <b>${fmt(s.held_psnr[j])} dB</b> PSNR against this photograph (measured by the NumPy renderer at ${META.img} px). The difference is this page's own render against the photograph: the brighter, the further off.`;
  drawMini();
}
function drawMini() {
  const id = META.held_ids[held.j], c = META.cams[id], { r, d, f, C } = camBasis(c);
  miniCam.position.copy(C);
  miniCam.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, d.clone().negate(), f.clone().negate()));
  miniCam.updateMatrixWorld();
  if (dirtyParams) { buildParams(shownT); dirtyParams = false; }
  const vis = camGroups.map((g) => g.visible); camGroups.forEach((g) => (g.visible = false));
  fillSorted(miniCam);
  splatMat.uniforms.uViewport.value.set(META.img, META.img); splatMat.uniforms.uDilWorld.value = 0;
  renderer.setRenderTarget(rt); renderer.clear(); renderer.render(scene, miniCam); renderer.setRenderTarget(null);
  renderer.readRenderTargetPixels(rt, 0, 0, META.img, META.img, pix);
  camGroups.forEach((g, i) => (g.visible = vis[i]));
  const n = META.img;
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const s = ((n - 1 - y) * n + x) * 4, t = (y * n + x) * 4; imgData.data[t] = pix[s]; imgData.data[t + 1] = pix[s + 1]; imgData.data[t + 2] = pix[s + 2]; imgData.data[t + 3] = 255; }
  mctx.putImageData(imgData, 0, 0);
  if (gtData) { // live |photo - Gaussians|, mean over RGB, x4, on the celadon ramp (what the PSNR above summarises)
    const g = gtData.data, D = diffData.data;
    for (let i = 0; i < n * n; i++) {
      const e = (Math.abs(imgData.data[i * 4] - g[i * 4]) + Math.abs(imgData.data[i * 4 + 1] - g[i * 4 + 1]) + Math.abs(imgData.data[i * 4 + 2] - g[i * 4 + 2])) / (3 * 255);
      const v = Math.min(1, 4 * e); D[i * 4] = 159 * v + 96 * v * v; D[i * 4 + 1] = 216 * v + 39 * v * v; D[i * 4 + 2] = 180 * v + 75 * v * v; D[i * 4 + 3] = 255;
    }
    dctx.putImageData(diffData, 0, 0);
  }
  dirtySort = true; requestDraw();
}
function heldOn() {
  if (held.on) return;
  held.on = true; held.j = (held.j + 1) % META.held_ids.length;
  held.saved = { az: view.az, el: view.el, dist: view.dist, k: view.dist / baseDist }; stopSpin();
  const c = META.cams[META.held_ids[held.j]];
  phSlots.forEach((el, k) => el.classList.toggle('shown', k === META.held_ids[held.j]));
  $('heldBtn').setAttribute('aria-pressed', 'true'); $('compare').classList.add('on'); document.body.classList.add('comparing'); updateCompare();
  const upright = innerWidth < 721 && !SIDE.matches;
  if (upright) resize();   // on an upright phone the card (now filled) bounds the stone's band
  glideTo(THREE.MathUtils.degToRad(c.az), THREE.MathUtils.degToRad(c.el), upright ? baseDist * held.saved.k : Math.max(view.dist, baseDist * 1.25), 600);
}
function heldOff() {
  if (!held.on) return; held.on = false;
  phSlots.forEach((el) => el.classList.remove('shown'));
  $('heldBtn').setAttribute('aria-pressed', 'false'); $('compare').classList.remove('on'); document.body.classList.remove('comparing', 'nostone');
  if (innerWidth < 721 && !SIDE.matches) { resize(); if (held.saved) held.saved.dist = baseDist * held.saved.k; }
  if (held.saved) glideTo(held.saved.az, held.saved.el, held.saved.dist, 600);
}
$('heldBtn').addEventListener('click', () => (held.on ? heldOff() : heldOn()));
document.addEventListener('keydown', (e) => { if ((e.key === 'h' || e.key === 'H') && !e.repeat && !e.metaKey && !e.ctrlKey && !e.altKey) heldOn(); });
document.addEventListener('keyup', (e) => { if (e.key === 'h' || e.key === 'H') heldOff(); });

// ------------------------------------------------------------ how it works (numbers from the log)
const s0 = META.snaps[0];
const rows = META.snaps.filter((s, i) => [0, 3, 5, 9, last].includes(i) || i === last).map((s) => `<tr><td>${int(s.step)}</td><td>${fmt(s.train_mean)}</td><td>${fmt(s.held_mean)}</td><td>${int(s.alive)}</td></tr>`).join('');
$('howBody').innerHTML = `
<h3>The photographs</h3>
<p>The stone is not real: it is a signed distance function, a squashed sphere pushed around by noise, with a quartz band painted through it. A small NumPy ray marcher renders it at ${META.img}&times;${META.img} px from sixteen known cameras. Twelve, in three rings of four, are the training set. Four sit between them and are never shown to the fit.</p>
<h3>The Gaussians</h3>
<p>${int(N)} Gaussians start as a cloud of faint, nearly grey blobs in a ball around the origin. Each has a position, three scales, a rotation (a quaternion), a colour and an opacity: ${F} numbers. No Gaussian is ever added (there is no densification), so every frame of the scrubber shows the same ${int(N)} Gaussians moving.${META.config.prune_schedule ? ` Every ${int(META.config.prune_schedule[1])} steps from step ${int(META.config.prune_schedule[0])} to step ${int(META.config.prune_schedule[2])}, Gaussians below opacity ${META.config.prune} are switched off for good: ${int(META.pruned[META.pruned.length - 1][1])} in total. They fade out in the scrubber instead of vanishing.` : ''}${META.config.opacity_reset ? ` Separately, every opacity is cut back once, at step ${int(META.config.opacity_reset)}${(META.pruned.filter((p) => p[0] < META.config.opacity_reset).pop() || [0, 0])[1] ? `, by which point ${int(META.pruned.filter((p) => p[0] < META.config.opacity_reset).pop()[1])} Gaussians had already been switched off` : ''}. 3DGS resets opacity periodically instead, every 3,000 iterations, along with its densification.` : ''}</p>
<h3>Splatting</h3>
<p>To draw one camera, each Gaussian's 3D covariance is projected to a 2D ellipse (the EWA splatting projection of Zwicker, Pfister, van Baar &amp; Gross, 2001), the ellipses are sorted front to back, and every pixel blends them in order, as in Kerbl, Kopanas, Leimk&uuml;hler &amp; Drettakis, &ldquo;3D Gaussian Splatting for Real-Time Radiance Field Rendering&rdquo;, ACM TOG 42(4) (SIGGRAPH 2023):</p>
<p class="eq">C = &Sigma;<sub>i</sub> c<sub>i</sub> &alpha;<sub>i</sub> &prod;<sub>j&lt;i</sub> (1 &minus; &alpha;<sub>j</sub>)</p>
<p>The same rule runs twice: once in NumPy for training, and once in this page's shader, which draws each Gaussian as an alpha-blended screen-space ellipse, re-sorted when the view turns.</p>
<h3>Learning</h3>
<p>The loss is the L1 difference between a render and a photograph, over a random solid background each step so that empty space has to become transparent. Every gradient, through the blending, the ellipse and the rotation, is derived by hand. On a tiny case they match finite differences to a relative error of ${META.grad_worst.toExponential(1)}. Adam takes ${int(META.config.steps)} steps, one random training photo each, in ${Math.round(META.seconds / 60)} minutes on two shared CPU cores.</p>
<p>PSNR is measured over the whole ${META.img}&nbsp;px image on its black background; the stone covers only about 45% of it. Counting stone pixels only, step 3,000 scores 31.9&nbsp;dB on the training views and 27.2&nbsp;dB on the held-out views.</p>
<table><tr><th>step</th><th>train dB</th><th>held-out dB</th><th>visible</th></tr>${rows}</table>
<h3>What it is not</h3>
<p>Colours do not depend on view direction (the lighting is baked into the stone), the photographs are synthetic and perfectly posed, and the images are tiny. Real 3DGS starts from structure-from-motion points, grows and prunes millions of Gaussians, and adds an SSIM term to the loss. This is the core idea at a small size.</p>
<p>The few blobs floating beside the stone are an honest failure: they sit outside the frame of most of the twelve cameras. Where one does fall inside a camera's frame, the stone is usually in front of it and hides it (187 of 228 such sightings); in 38 the blob sits in front of the stone, and only 3 are against empty background. So almost no photograph contradicts them, mainly because the stone hides them. More cameras, or wider ones, should carve them away.</p>`;

// ------------------------------------------------------------ frame loop (draws only when something changed)
let raf = 0, lastT = performance.now();
function requestDraw() { if (!raf) raf = requestAnimationFrame(frame); }
const ease = (x) => x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
function frame(now) {
  raf = 0; const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
  let more = false;
  if (tween) { const u = Math.min(1, (now - tween.t0) / tween.ms); shownT = tween.from + (tween.to - tween.from) * ease(u); dirtyParams = true; if (u >= 1) tween = null; else more = true; }
  if (anim) { const u = Math.min(1, (now - anim.t0) / anim.ms), e = ease(u); for (const k of ['az', 'el', 'dist']) view[k] = anim.from[k] + (anim.to[k] - anim.from[k]) * e; dirtySort = true; if (u >= 1) anim = null; else more = true; }
  if (autoSpin && !held.on && !anim) { view.az += dt * 0.12; dirtySort = true; more = true; }
  if (dirtyParams) { buildParams(shownT); dirtyParams = false; dirtySort = true; if (held.on && !tween) drawMini(); }
  placeCamera(camera, view.az, view.el, view.dist);
  if (dirtySort) { fillSorted(camera); dirtySort = false; }
  const cpos = camera.position; camGroups.forEach((g) => { const behind = g.userData.C.dot(cpos) < 0; g.renderOrder = behind ? -1 : 1; g.children.forEach((c) => (c.renderOrder = g.renderOrder)); });
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  splatMat.uniforms.uViewport.value.copy(size); splatMat.uniforms.uDilWorld.value = DIL_WORLD;
  renderer.render(scene, camera);
  if (more) requestDraw();
}
function resize() {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  // centre the stone in the space left between the title and the controls (matters on phones)
  const hdr = document.querySelector('header').getBoundingClientRect().bottom, bot = document.querySelector('.dock').getBoundingClientRect().top;
  const cmp = $('compare');
  // phones, comparing: the card sits under the title, and the stone takes the band between the card and the dock
  const side = SIDE.matches, upright = innerWidth < 721 && !side;
  if (upright && held.on) { cmp.style.top = Math.round(hdr + 10) + 'px'; } else cmp.style.top = '';
  const top = upright && held.on ? cmp.getBoundingClientRect().bottom : hdr;
  const phone = upright && bot > top;
  // a short phone, comparing: under 60 px are left between the card and the dock, so the 3D view steps aside
  // (the card shows the stone three times; its images are rendered off screen and are not affected)
  document.body.classList.toggle('nostone', upright && held.on && bot - top < 60);
  // a phone on its side: the stone goes in the space right of the left column (title, curve, readout), above the buttons
  const tv = Math.tan(THREE.MathUtils.degToRad(META.fov) / 2);
  let dx = 0, dy = phone ? Math.round(innerHeight / 2 - (top + bot) / 2) : 0, bandW = 0, bandH = 0;
  if (side) {
    const tr = document.querySelector('.train').getBoundingClientRect(), dk = document.querySelector('.dock').getBoundingClientRect();
    const t0 = document.querySelector('.meta').getBoundingClientRect().bottom + 6, b0 = document.querySelector('.side').getBoundingClientRect().top - 8;
    const l0 = tr.right + 12, r0 = dk.right;
    bandW = Math.max(1, r0 - l0); bandH = Math.max(1, b0 - t0);
    dx = Math.round(innerWidth / 2 - (l0 + r0) / 2); dy = Math.round(innerHeight / 2 - (t0 + b0) / 2);
  }
  if (dx || dy) camera.setViewOffset(innerWidth, innerHeight, dx, dy, innerWidth, innerHeight); else camera.clearViewOffset();
  camera.updateProjectionMatrix();
  // on a phone the stone must also fit the height of that band (else the dock covers it): step back if needed
  let nb = fitDist();
  if (phone) nb = Math.max(nb, FIT.band / (tv * Math.max(0.2, (bot - top) / innerHeight)));
  if (side) nb = Math.max(nb, FIT.band / (tv * Math.max(0.2, bandH / innerHeight)), FIT.band / (tv * camera.aspect * Math.max(0.2, bandW / innerWidth)));
  view.dist = view.dist ? view.dist * nb / baseDist : nb; baseDist = nb;
  dirtySort = true; requestDraw();
}
window.addEventListener('resize', resize);
resize();

// first visit: show the cloud, then let it train once (unless reduced motion asks for stillness)
if (reduced) { goTo(last, 0); } else { goTo(0, 0); setTimeout(() => { if (!userTouched && !playing && +slider.value === 0) play(); }, 900); }
// the readout and the web fonts change the dock's and header's height: measure the free band again once they are in
requestAnimationFrame(resize);
if (document.fonts && document.fonts.ready) document.fonts.ready.then(resize);

// test hooks
window.__splat = {
  ready: true, N, S, setDil: (k) => { DIL_WORLD = k * 0.3 * Math.pow(META.dist / META.focal, 2); frame(performance.now()); },
  setStep: (k) => { stopPlay(); goTo(k, 0); },
  stopSpin, cams: (on) => { if (on !== camsOn) camsBtn.click(); },
  held: (on) => (on ? heldOn() : heldOff()),
  renderNow: () => { anim = null; tween = null; frame(performance.now()); },
  view, settle: () => { if (anim) { for (const k of ['az', 'el', 'dist']) view[k] = anim.to[k]; anim = null; } if (tween) { shownT = tween.to; tween = null; dirtyParams = true; } frame(performance.now()); },
  miniPSNR: () => { // live check against the photograph (for the test only; the page shows the logged number)
    const id = META.held_ids[held.j], n = META.img, cv2 = document.createElement('canvas'); cv2.width = cv2.height = n * 4;
    const c2 = cv2.getContext('2d'); c2.drawImage(atlasImg, 0, 0); const g = c2.getImageData((id % 4) * n, Math.floor(id / 4) * n, n, n).data;
    const m = mctx.getImageData(0, 0, n, n).data; let se = 0; for (let i = 0; i < n * n; i++) for (let ch = 0; ch < 3; ch++) { const e = (m[i * 4 + ch] - g[i * 4 + ch]) / 255; se += e * e; }
    return -10 * Math.log10(se / (n * n * 3));
  },
};
})();
