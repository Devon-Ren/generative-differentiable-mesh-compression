# viewer-v3: fix — orbit / spin / drag now turn around the OBJECT CENTRE (2026-11-04 15:35)

v2 bug (user report): the orbit centre was "0.3 × bbox diagonal in front of the camera", not the object, so spin and drag wobbled around a point in space.
v3: a fixed world-space turntable target = bbox centre (set at load, moved only by pan), yaw around the world up axis (mean of the training cameras' up vectors, else +z), pitch around the camera right axis; dolly moves toward the centre and never through it; `fit object` / R re-centres. Verified in headless Chromium on an off-centre test object with ring cameras: the object's screen centroid stays within 1 px through spin, drag orbit and camera stepping; dolly only changes its size.
Everything else (bg black/white/grey, wireframe, cameras, benchmark) = v2. The .msv files from viewer-v2 §1 are unchanged; only the HTML is new.

## §1 HPC: write viewer-v3.html (one paste; no export needed if viewer-v2/msv already exists)

```bash
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v3; mkdir -p $VW
cat > $VW/viewer-v3.html <<'HTMLEOF_V3'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>MeshSplat viewer v3</title>
<style>
:root { --bg:#111; --fg:#eee; --mut:#9a9a9a; --acc:#1baf7a; }
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:13px/1.4 Inter,system-ui,sans-serif;overflow:hidden}
#c{display:block;position:absolute;left:0;top:0;width:100%;height:100%;touch-action:none}
#ui{position:absolute;left:0;top:0;padding:8px 10px;background:rgba(0,0,0,.55);max-width:100%;box-sizing:border-box}
#ui .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:2px 0}
button,select,input[type=file]{font:inherit;background:#222;color:var(--fg);border:1px solid #444;border-radius:4px;padding:3px 8px}
button:active{background:#333}
#stats{font-variant-numeric:tabular-nums;color:var(--fg)}
#stats b{color:var(--acc)}
.mut{color:var(--mut)}
#drop{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--mut);pointer-events:none;font-size:15px;text-align:center;padding:20px}
#bar{position:absolute;left:0;bottom:0;height:3px;background:var(--acc);width:0}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="drop">Open a .msv file (button top-left), or add ?file=name.msv to the URL</div>
<div id="bar"></div>
<div id="ui">
  <div class="row">
    <input type="file" id="file" accept=".msv">
    <select id="res"><option value="dataset">dataset resolution</option><option value="dpr">window × DPR</option><option value="1x">window × 1</option></select>
    <select id="deg"><option value="3">SH degree 3</option><option value="2">SH degree 2</option><option value="1">SH degree 1</option><option value="0">SH degree 0</option></select>
    <select id="bg"><option value="black">black bg</option><option value="white">white bg</option><option value="grey">grey bg</option></select>
    <label><input type="checkbox" id="wire"> wireframe</label>
    <button id="fit">fit object</button>
    <button id="prev">◀ cam</button><button id="next">cam ▶</button>
    <label><input type="checkbox" id="spin"> spin</label>
    <button id="bench">benchmark (300 frames)</button>
  </div>
  <div class="row" id="stats"><span class="mut">no file loaded</span></div>
  <div class="row mut" id="info"></div>
  <div class="row" id="benchline" style="user-select:text;-webkit-user-select:text;word-break:break-all"></div>
  <div class="row mut" id="help">drag: orbit around the object · wheel / pinch: dolly · shift-drag / middle: pan · W A S D Q E: move · R: reset view · viewer v3 (orbit target = object centre)</div>
</div>
<script>
'use strict';
const $ = id => document.getElementById(id);
const canvas = $('c'); const gl = canvas.getContext('webgl2', {antialias:false, powerPreference:'high-performance', preserveDrawingBuffer:false});
if (!gl) { $('drop').textContent = 'WebGL2 not available in this browser'; throw new Error('no webgl2'); }
const dbg = gl.getExtension('WEBGL_debug_renderer_info');
const gpuName = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'unknown GPU';
const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
const CODES_PER_ROW = 256, TEX_W = CODES_PER_ROW * 12;

// ---------- shaders (SH evaluation at the vertex, Gouraud interpolation by the rasterizer; 3DGS SH convention) ----------
function vertexShaderSource() {
  const C1 = 0.4886025119029199, C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396];
  const C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435];
  const basis = [null,
    `(${-C1}*y)`, `(${C1}*z)`, `(${-C1}*x)`,
    `(${C2[0]}*xy)`, `(${C2[1]}*yz)`, `(${C2[2]}*(2.0*zz-xx-yy))`, `(${C2[3]}*xz)`, `(${C2[4]}*(xx-yy))`,
    `(${C3[0]}*y*(3.0*xx-yy))`, `(${C3[1]}*xy*z)`, `(${C3[2]}*y*(4.0*zz-xx-yy))`, `(${C3[3]}*z*(2.0*zz-3.0*xx-3.0*yy))`,
    `(${C3[4]}*x*(4.0*zz-xx-yy))`, `(${C3[5]}*z*(xx-yy))`, `(${C3[6]}*x*(xx-3.0*yy))`];
  const comp = ['x','y','z','w'];
  const coef = k => { const o = 3*(k-1); return [0,1,2].map(ch => { const i = o+ch; return `t${Math.floor(i/4)}.${comp[i%4]}`; }); };
  let fetch = ''; for (let j = 0; j < 12; j++) fetch += `vec4 t${j} = tx(uBank, aIdx.x, ${j});\n`;
  let rfetch = ''; for (let j = 0; j < 12; j++) rfetch += `t${j} += tx(uRBank, aIdx.y, ${j});\n`;
  let acc = '';
  for (let k = 1; k <= 15; k++) { const deg = k < 4 ? 1 : k < 9 ? 2 : 3; const c = coef(k); acc += `if (uDeg >= ${deg}) c += ${basis[k]} * vec3(${c[0]}, ${c[1]}, ${c[2]});\n`; }
  return `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
layout(location=0) in vec3 aPos; layout(location=1) in vec3 aDc; layout(location=2) in uvec2 aIdx;
uniform mat4 uViewProj; uniform vec3 uCam; uniform sampler2D uBank; uniform sampler2D uRBank; uniform int uDeg;
out vec3 vColor;
vec4 tx(sampler2D s, uint code, int j) { return texelFetch(s, ivec2(int(code & ${CODES_PER_ROW-1}u) * 12 + j, int(code >> 8)), 0); }
void main() {
  vec3 d = normalize(aPos - uCam);
  float x = d.x, y = d.y, z = d.z, xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
  vec3 c = 0.28209479177387814 * aDc;
  ${fetch}
  if (aIdx.y != 0xFFFFFFFFu) { ${rfetch} }
  ${acc}
  vColor = max(c + 0.5, 0.0);
  gl_Position = uViewProj * vec4(aPos, 1.0);
}`;
}
const FS = `#version 300 es
precision mediump float; in vec3 vColor; uniform int uWire; out vec4 o; void main() { o = uWire == 1 ? vec4(0.15, 0.85, 1.0, 1.0) : vec4(vColor, 1.0); }`;
function compile(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s)); return s; }
const prog = gl.createProgram(); gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertexShaderSource())); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
const U = {}; for (const n of ['uViewProj', 'uCam', 'uBank', 'uRBank', 'uDeg', 'uWire']) U[n] = gl.getUniformLocation(prog, n);

// ---------- small matrix helpers (column-major 4x4) ----------
const m4 = {
  mul(a, b) { const o = new Float32Array(16); for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+r] * b[c*4+k]; o[c*4+r] = s; } return o; },
};
const v3 = { sub:(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], add:(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], scale:(a,s)=>[a[0]*s,a[1]*s,a[2]*s],
  dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], len:a=>Math.hypot(a[0],a[1],a[2]), norm(a){const l=Math.hypot(a[0],a[1],a[2])||1;return [a[0]/l,a[1]/l,a[2]/l];},
  cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]] };
// rotate vector v about unit axis u by angle t (Rodrigues)
function rot(v, u, t) { const c = Math.cos(t), s = Math.sin(t), d = v3.dot(u, v); const cr = v3.cross(u, v);
  return [v[0]*c + cr[0]*s + u[0]*d*(1-c), v[1]*c + cr[1]*s + u[1]*d*(1-c), v[2]*c + cr[2]*s + u[2]*d*(1-c)]; }

// ---------- state ----------
let mesh = null;          // {V,F,K,R,header,vao,ebo,bank,rbank}
let cam = null;           // {pos:[3], R:[right,down,fwd] world-space columns (COLMAP: x right, y down, z forward), fx, fy, w, h, target}
let camIndex = 0, diag = 1, target = [0, 0, 0], worldUp = [0, 0, 1];   // orbit target = object (bbox) centre; worldUp = mean camera up, else +z
let frames = [], benchRunning = false;

function setStatus(html) { $('stats').innerHTML = html; }
function setInfo(html) { $('info').innerHTML = html; }

function makeTexture(data, rows) {
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, TEX_W, rows, 0, gl.RGBA, gl.FLOAT, data); return tex;
}
function bankTexture(bank45, K) {  // K x 45 floats -> rows of 256 codes x 12 RGBA texels (48 floats, 3 pad)
  const rows = Math.max(1, Math.ceil(K / CODES_PER_ROW));
  if (rows > maxTex) throw new Error(`codebook of ${K} codes needs ${rows} texture rows > MAX_TEXTURE_SIZE ${maxTex} on this GPU`);
  const data = new Float32Array(TEX_W * rows * 4);
  for (let k = 0; k < K; k++) { const base = ((k >> 8) * TEX_W + (k & 255) * 12) * 4; for (let i = 0; i < 45; i++) data[base + i] = bank45[k * 45 + i]; }
  return makeTexture(data, rows);
}

function parseMSV(buf) {
  const dv = new DataView(buf); if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'MSV1') throw new Error('not an MSV1 file');
  const hlen = dv.getUint32(4, true); const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
  let off = 8 + hlen; const blobs = {};
  for (const b of header.blobs) { const n = b.shape.reduce((a, c) => a * c, 1); const src = off % 4 ? buf.slice(off, off + b.bytes) : buf, o = off % 4 ? 0 : off; blobs[b.name] = b.dtype === 'float32' ? new Float32Array(src, o, n) : new Uint32Array(src, o, n); off += b.bytes; }
  return { header, blobs };
}

function loadMesh(buf, name) {
  const { header, blobs } = parseMSV(buf);
  if (mesh) { gl.deleteVertexArray(mesh.vao); gl.deleteTexture(mesh.bank); gl.deleteTexture(mesh.rbank); if (mesh.wbo) gl.deleteBuffer(mesh.wbo); }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const attr = (loc, data, size, integer) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc); if (integer) gl.vertexAttribIPointer(loc, size, gl.UNSIGNED_INT, 0, 0); else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); };
  attr(0, blobs.xyz, 3, false); attr(1, blobs.dc, 3, false);
  const idx = new Uint32Array(header.V * 2); for (let i = 0; i < header.V; i++) { idx[2*i] = blobs.mapping[i]; idx[2*i+1] = blobs.residual[i]; }
  attr(2, idx, 2, true);
  const ebo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, blobs.faces, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  const bank = bankTexture(blobs.bank, header.K); const rbank = bankTexture(header.R ? blobs.residual_bank : new Float32Array(45), Math.max(1, header.R));
  mesh = { header, vao, ebo, bank, rbank, V: header.V, F: header.F, name, faces: blobs.faces, wbo: null };
  const [lo, hi] = header.bbox; diag = v3.len(v3.sub(hi, lo)); target = v3.scale(v3.add(lo, hi), 0.5);
  worldUp = [0, 0, 1];
  if (header.cameras.length) { let u = [0, 0, 0]; for (const c of header.cameras) u = v3.add(u, [-c.rotation[0][1], -c.rotation[1][1], -c.rotation[2][1]]);   // camera up = -(column 1 of C2W)
    if (v3.len(u) > 1e-6) worldUp = v3.norm(u); }
  camIndex = 0; setCamera(0);
  const m = header.meta || {};
  setInfo(`${header.label} · V ${header.V.toLocaleString()} · F ${header.F.toLocaleString()} · K ${header.K} R ${header.R} · source ${m.source || '?'} ${(m.file_bytes/1e6).toFixed(2)} MB` +
          (m.decode_total_s != null ? ` · decode ${m.decode_total_s} s` : '') + ` · cameras ${header.cameras.length} · ${gpuName}`);
  $('drop').style.display = 'none';
}

function fitObject() { // look at the bbox centre from a 3/4 view at 1.6 x bbox diagonal; orbit target = the centre (R key / 'fit object')
  const [lo, hi] = mesh.header.bbox; target = v3.scale(v3.add(lo, hi), 0.5);
  // 3/4 view: a horizontal direction perpendicular to worldUp, tilted 30 deg down
  let h = v3.cross(worldUp, [1, 0, 0]); if (v3.len(h) < 1e-3) h = v3.cross(worldUp, [0, 1, 0]); h = v3.norm(h); const h2 = v3.norm(v3.cross(worldUp, h));
  const fw = v3.norm(v3.add(v3.scale(v3.norm(v3.add(h, h2)), Math.cos(Math.PI / 6)), v3.scale(worldUp, -Math.sin(Math.PI / 6))));
  const pos = v3.sub(target, v3.scale(fw, 1.6 * diag));
  let r = v3.norm(v3.cross(worldUp, fw)); if (!v3.len(r)) r = [1, 0, 0]; const d = v3.norm(v3.cross(fw, r));
  cam = { pos, R: [r, d, fw], fx: 1100, fy: 1100, w: 1200, h: 900, dist: 1.6 * diag, name: 'fit' };
}
function setCamera(i) {
  const cams = mesh.header.cameras;
  if (!cams.length) { fitObject(); return; }
  i = ((i % cams.length) + cams.length) % cams.length; camIndex = i; const c = cams[i];
  // rotation is C2W: column j = camera axis j in world (x right, y down, z forward)
  const R = [[c.rotation[0][0], c.rotation[1][0], c.rotation[2][0]], [c.rotation[0][1], c.rotation[1][1], c.rotation[2][1]], [c.rotation[0][2], c.rotation[1][2], c.rotation[2][2]]];
  cam = { pos: c.position.slice(), R, fx: c.fx, fy: c.fy, w: c.width, h: c.height, dist: v3.len(v3.sub(target, c.position)), name: c.name || String(c.id) };
}

function resize() {
  const mode = $('res').value; const dpr = mode === 'dpr' ? (window.devicePixelRatio || 1) : 1;
  let w, h;
  if (mode === 'dataset' && cam) { w = cam.w; h = cam.h; } else { w = Math.round(window.innerWidth * dpr); h = Math.round(window.innerHeight * dpr); }
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  // letterbox the drawing buffer into the window, preserving aspect
  const s = Math.min(window.innerWidth / w, window.innerHeight / h); canvas.style.width = (w * s) + 'px'; canvas.style.height = (h * s) + 'px';
  canvas.style.left = ((window.innerWidth - w * s) / 2) + 'px'; canvas.style.top = ((window.innerHeight - h * s) / 2) + 'px';
}

function viewProj() {
  // camera-space c = R^T (p - pos); clip.x = (2 fx / W) c.x, clip.y = -(2 fy / H) c.y (image y down -> GL y up), clip.w = c.z
  const W = canvas.width, H = canvas.height; const sx = 2 * cam.fx / cam.w, sy = 2 * cam.fy / cam.h;   // keep the dataset FOV whatever the buffer size
  const n = 1e-3 * diag, f = 20 * diag, A = (f + n) / (f - n), B = -2 * f * n / (f - n);
  const [r, d, fw] = cam.R; const p = cam.pos;
  const view = new Float32Array([r[0], d[0], fw[0], 0, r[1], d[1], fw[1], 0, r[2], d[2], fw[2], 0, -v3.dot(r, p), -v3.dot(d, p), -v3.dot(fw, p), 1]);
  // aspect fix when the buffer aspect differs from the dataset aspect: scale x so that the horizontal FOV follows the buffer
  const ax = (W / H) / (cam.w / cam.h);
  const proj = new Float32Array([sx / ax, 0, 0, 0, 0, -sy, 0, 0, 0, 0, A, 1, 0, 0, B, 0]);
  return m4.mul(proj, view);
}

function draw() {
  gl.viewport(0, 0, canvas.width, canvas.height); const bgc = { black: [0, 0, 0], white: [1, 1, 1], grey: [0.5, 0.5, 0.5] }[$('bg').value]; gl.clearColor(bgc[0], bgc[1], bgc[2], 1); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!mesh) return;
  gl.useProgram(prog); gl.bindVertexArray(mesh.vao);
  gl.uniformMatrix4fv(U.uViewProj, false, viewProj()); gl.uniform3fv(U.uCam, cam.pos); gl.uniform1i(U.uDeg, parseInt($('deg').value));
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, mesh.bank); gl.uniform1i(U.uBank, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mesh.rbank); gl.uniform1i(U.uRBank, 1);
  const wire = $('wire').checked; if (wire) { gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1.0, 1.0); } else gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.uniform1i(U.uWire, 0); gl.drawElements(gl.TRIANGLES, mesh.F * 3, gl.UNSIGNED_INT, 0);
  if (wire) { // triangle edges on top of the fill (fill pushed back by polygon offset so the lines win the z-test)
    if (!mesh.wbo) { const e = new Uint32Array(mesh.F * 6), f = mesh.faces; for (let i = 0; i < mesh.F; i++) { const a = f[3*i], b = f[3*i+1], c = f[3*i+2]; e.set([a, b, b, c, c, a], 6 * i); }
      mesh.wbo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.wbo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, e, gl.STATIC_DRAW); }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.wbo); gl.uniform1i(U.uWire, 1); gl.disable(gl.POLYGON_OFFSET_FILL); gl.depthFunc(gl.LEQUAL);
    gl.drawElements(gl.LINES, mesh.F * 6, gl.UNSIGNED_INT, 0); gl.depthFunc(gl.LESS); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.ebo);
  }
  gl.bindVertexArray(null);
}

let last = performance.now();
function loop(t) {
  if (!benchRunning) {
    resize();
    if ($('spin').checked && cam) orbit(0.006, 0);
    draw();
    frames.push(t); while (frames.length && frames[0] < t - 2000) frames.shift();
    const fps = frames.length > 1 ? (frames.length - 1) * 1000 / (frames[frames.length - 1] - frames[0]) : 0;
    if (mesh) setStatus(`<b>${fps.toFixed(1)} fps</b> (display, vsync-bound) · ${canvas.width}×${canvas.height} · cam ${camIndex + 1}/${mesh.header.cameras.length || 1} ${cam.name}`);
  }
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// benchmark: N draws, forced to complete with a 1-pixel readback; ms per frame is then GPU-bound, not vsync-bound
async function benchmark() {
  if (!mesh || benchRunning) return; benchRunning = true; resize();
  const N = 300, cams = mesh.header.cameras.length || 1, saved = camIndex, px = new Uint8Array(4);
  const times = [];
  for (let i = 0; i < N; i++) {
    if (cams > 1 && i % 10 === 0) setCamera(saved + Math.floor(i / 10));
    const t0 = performance.now(); draw(); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px); times.push(performance.now() - t0);
    if (i % 25 === 0) { setStatus(`benchmark ${i}/${N} …`); $('bar').style.width = (100 * i / N) + '%'; await new Promise(r => requestAnimationFrame(r)); }
  }
  $('bar').style.width = '0'; setCamera(saved);
  times.sort((a, b) => a - b); const mean = times.reduce((a, b) => a + b, 0) / N, med = times[N >> 1], p95 = times[Math.floor(0.95 * N)];
  const line = `${mesh.header.label} | ${canvas.width}x${canvas.height} | F ${mesh.F} | mean ${mean.toFixed(2)} ms = ${(1000 / mean).toFixed(0)} fps | median ${med.toFixed(2)} ms | p95 ${p95.toFixed(2)} ms | SH deg ${$('deg').value} | ${gpuName} | ${navigator.userAgent}`;
  console.log('BENCH', line); benchRunning = false;
  setStatus(`<b>${(1000 / mean).toFixed(0)} fps</b> GPU-bound (mean ${mean.toFixed(2)} ms, median ${med.toFixed(2)}, p95 ${p95.toFixed(2)}) at ${canvas.width}×${canvas.height} · <button id="copy">copy line</button>`);
  $('benchline').textContent = 'BENCH ' + line;   // selectable on phones (clipboard API needs https)
  $('copy').onclick = () => navigator.clipboard && navigator.clipboard.writeText('BENCH ' + line);
}

// ---------- controls ----------
function orbit(dyaw, dpitch) { // turntable: rotate the camera about the OBJECT CENTRE (target): yaw around worldUp, pitch around the camera's right axis
  let rel = v3.sub(cam.pos, target), R = cam.R;
  rel = rot(rel, worldUp, -dyaw); R = R.map(a => rot(a, worldUp, -dyaw));
  const right = R[0]; rel = rot(rel, right, -dpitch); R = R.map(a => rot(a, right, -dpitch));
  cam.pos = v3.add(target, rel); cam.R = R; cam.dist = v3.len(rel);
}
function dolly(f) { // move toward / away from the object centre; never through it
  const rel = v3.sub(cam.pos, target); const dist = v3.len(rel); const nd = Math.max(0.02 * diag, dist * (1 - f));
  cam.pos = v3.add(target, v3.scale(rel, nd / dist)); cam.dist = nd;
}
function pan(dx, dy) { const [r, d] = cam.R; const s = cam.dist * 0.002; const mv = v3.add(v3.scale(r, -dx * s), v3.scale(d, -dy * s)); cam.pos = v3.add(cam.pos, mv); target = v3.add(target, mv); }
let drag = null, pinch = null;
canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, id: e.pointerId, shift: e.shiftKey }; });
canvas.addEventListener('pointermove', e => { if (!drag || !cam || e.pointerId !== drag.id) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.shift || e.buttons === 4) pan(dx, dy); else orbit(dx * 0.005, dy * 0.005); });
canvas.addEventListener('pointerup', () => drag = null); canvas.addEventListener('pointercancel', () => drag = null);
canvas.addEventListener('wheel', e => { if (cam) dolly(-Math.sign(e.deltaY) * 0.1); e.preventDefault(); }, { passive: false });
canvas.addEventListener('touchstart', e => { if (e.touches.length === 2) { drag = null; pinch = { d: Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY), x: (e.touches[0].clientX + e.touches[1].clientX) / 2, y: (e.touches[0].clientY + e.touches[1].clientY) / 2 }; } }, { passive: true });
canvas.addEventListener('touchmove', e => { if (e.touches.length === 2 && pinch && cam) { const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
  const x = (e.touches[0].clientX + e.touches[1].clientX) / 2, y = (e.touches[0].clientY + e.touches[1].clientY) / 2; dolly((d - pinch.d) * 0.005); pan(x - pinch.x, y - pinch.y); pinch = { d, x, y }; e.preventDefault(); } }, { passive: false });
canvas.addEventListener('touchend', () => pinch = null);
window.addEventListener('keydown', e => { if (!cam) return; const s = 0.02 * diag; const [r, d, fw] = cam.R; const k = e.key.toLowerCase(); if (k === 'r' && mesh) { fitObject(); return; }
  const mv = { w: fw, s: v3.scale(fw, -1), a: v3.scale(r, -1), d: r, q: d, e: v3.scale(d, -1) }[k]; if (mv) cam.pos = v3.add(cam.pos, v3.scale(mv, s)); });
$('prev').onclick = () => mesh && setCamera(camIndex - 1); $('next').onclick = () => mesh && setCamera(camIndex + 1);
$('bench').onclick = benchmark; $('res').onchange = resize; $('fit').onclick = () => { if (mesh) fitObject(); };
$('file').onchange = e => { const f = e.target.files[0]; if (!f) return; setStatus('reading ' + f.name + ' …'); f.arrayBuffer().then(b => loadMesh(b, f.name)).catch(err => { setStatus('FAILED: ' + err.message); console.error(err); }); };
const q = new URLSearchParams(location.search).get('file');
if (q) { setStatus('fetching ' + q + ' …'); fetch(q).then(async r => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); const total = +r.headers.get('content-length') || 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (total) $('bar').style.width = (100 * got / total) + '%'; }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; } $('bar').style.width = '0'; return buf.buffer; })
  .then(b => loadMesh(b, q)).catch(err => { setStatus('FAILED: ' + err.message); console.error(err); }); }
</script>
</body>
</html>
HTMLEOF_V3
grep -c "viewer v3" $VW/viewer-v3.html; ls -l $VW/viewer-v3.html $AN/viewer-v2/msv/*.msv 2>/dev/null | head -12
```

## §2 laptop

```bash
scp 'torch:/scratch/zr2245/analysis/viewer-v3/viewer-v3.html' ~/msv2/ && ls -la ~/msv2
```

Open `~/msv2/viewer-v3.html` in Chrome, Choose File → the same `.msv` as before. Drag = orbit around the object, wheel = dolly, shift-drag = pan (moves the turntable centre with the view), R = re-centre, spin = turntable.
