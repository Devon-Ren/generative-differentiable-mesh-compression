# viewer-v1: standard-mesh web viewer for desktop / phone FPS + decode time (2026-10-25)

What it measures (Table 2 "deployability" columns, see plan/speed-direction-2026-10-25.md §4):
- **decode time**: asset.mslb -> standard mesh + per-vertex SH indices, on one CPU core of the GPU node (native decoders precompiled, so the number is pure decode). Printed by export_viewer_v1.py.
- **FPS** on a laptop GPU and on a phone: viewer.html draws the decoded mesh with the ordinary GPU pipeline (index buffer + vertex shader that evaluates the degree-3 SH from the two codebook textures, no sorting, no splatting). The benchmark button renders 300 frames at the dataset resolution through the training cameras and prints one `BENCH` line.

Tested here on a synthetic 40k-vertex asset in headless Chromium (render + benchmark + camera switch); the projection convention was checked numerically against the 3DGS cameras.json convention (x right, y down, z forward). First real-scene run is yours: if the picture looks wrong (mirrored / black), paste the BENCH line and a screenshot and I fix the viewer.

Files installed by §1 (all under `/scratch/zr2245/analysis/viewer-v1/`, nothing in the repos is touched):
- `export_viewer_v1.py` — decodes one asset.mslb (or reads an original checkpoint .pt) and writes `<scene>.msv`
- `viewer.html` — single-file WebGL2 viewer (opens .msv via file picker or `?file=`)
- `export-all-v1.sh` — runs the export for the 11 λ6.2e-3 assets + Room λ1.85e-3, writes `decode-times-v1.txt`

GPU-hour cost: zero (CPU decode only, ~1 min per scene on the node you already hold).

## §1 install (GPU node, any shell; paste as one block)

```bash
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v1; mkdir -p $VW
cat > $VW/export_viewer_v1.py <<'PYEOF_V1'
"""export_viewer_v1.py (2026-10-25): decode a delivered asset.mslb (or read an original checkpoint .pt) and write the
compact viewer file <out>.msv for viewer.html (standard mesh + per-vertex SH via codebooks, standard GPU pipeline).

usage (inside the stage2 env, e.g. via vis_collapse/stage2_run.sh):
  python export_viewer_v1.py --stage2 /scratch/zr2245/shared-collapse-stage2-v2 --asset .../delivery/asset.mslb \
      --cameras /scratch/zr2245/room-vis-collapse/original/cameras.json --out room-lam0.0062.msv [--label room]
  python export_viewer_v1.py --stage2 ... --ckpt .../point_cloud_state_dict.pt --cameras ... --out room-original.msv

.msv layout: b'MSV1' | uint32 header_len | JSON header | blobs in header order (little endian):
  xyz float32 [V,3] | faces uint32 [F,3] | dc float32 [V,3] | mapping uint32 [V] | residual uint32 [V] (0xFFFFFFFF = none)
  | bank float32 [K,45] | residual_bank float32 [R,45]
Also prints decode wall time (read + rebuild, native decoders precompiled into --native-cache) for the paper's decode-time column."""
import argparse, json, sys, time, struct, tempfile
from pathlib import Path
import numpy as np

def arr(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)

def from_asset(path, stage2, native_cache):
    for sub in ('renderer', '', 'codec'):
        sys.path.insert(0, str(Path(stage2) / sub))
    import topology as tc, local_codec as lc, index_context as ic
    from delivery import restore
    cache = Path(native_cache); cache.mkdir(parents=True, exist_ok=True)
    lc.native(cache); tc.edge_native(cache); ic.native(cache)     # compile the three native decoders once, outside the timer
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wire = tc.read_asset(path, cache, tmp)                    # container -> faces -> XYZ -> SH indices (all streams)
    t_read = time.perf_counter() - t0
    state = restore(dict(wire))                                   # dequantise XYZ / DC, constant opacity
    t_all = time.perf_counter() - t0
    xyz = arr(state['triangles_points']).astype(np.float32).reshape(-1, 3)
    faces = arr(state['_triangle_indices']).astype(np.uint32).reshape(-1, 3)
    dc = arr(state['features_dc']).astype(np.float32).reshape(len(xyz), 3)
    if wire.get('app_schema') != 2 or np.asarray(wire.get('app_gain', ())).size or np.asarray(wire.get('app_correction', ())).size:
        raise SystemExit('FAILED: expected app_schema 2 without gains/corrections; got schema %r' % wire.get('app_schema'))
    bank = np.asarray(wire['app_bank'], np.float32).reshape(-1, 45)
    mapping = np.asarray(wire['app_mapping']).astype(np.uint32)
    rbank = np.asarray(wire['app_residual_bank'], np.float32).reshape(-1, 45)
    rows = np.asarray(wire['app_rows']).astype(np.int64); rmap = np.asarray(wire['app_residual_mapping']).astype(np.uint32)
    residual = np.full(len(xyz), 0xFFFFFFFF, np.uint32); residual[rows] = rmap
    meta = dict(source='asset', asset=str(path), file_bytes=Path(path).stat().st_size, decode_read_s=round(t_read, 3), decode_total_s=round(t_all, 3),
                opacity_const=float(wire.get('vertex_weight_const', 0.0)), opacity_floor=float(wire.get('opacity_floor', 0.999)))
    return xyz, faces, dc, mapping, residual, bank, rbank, meta

def from_ckpt(path):
    import torch
    t0 = time.perf_counter(); s = torch.load(path, map_location='cpu', weights_only=False); t_all = time.perf_counter() - t0
    g = lambda *n: next(s[k] for k in n if k in s)
    xyz = arr(g('triangles_points')).astype(np.float32).reshape(-1, 3)
    faces = arr(g('_triangle_indices')).astype(np.uint32).reshape(-1, 3)
    dc = arr(g('features_dc', '_features_dc')).astype(np.float32).reshape(len(xyz), 3)
    rest = arr(g('features_rest', '_features_rest')).astype(np.float32).reshape(len(xyz), 45)
    meta = dict(source='checkpoint', asset=str(path), file_bytes=Path(path).stat().st_size, decode_total_s=round(t_all, 3))
    return xyz, faces, dc, np.arange(len(xyz), dtype=np.uint32), np.full(len(xyz), 0xFFFFFFFF, np.uint32), rest, np.zeros((0, 45), np.float32), meta

def load_cameras(path, limit):
    if not path: return []
    cams = json.loads(Path(path).read_text()); out = []
    for c in cams[:limit]:
        out.append(dict(id=c.get('id'), name=c.get('img_name'), width=int(c['width']), height=int(c['height']), fx=float(c['fx']), fy=float(c['fy']),
                        position=[float(v) for v in c['position']], rotation=[[float(v) for v in r] for r in c['rotation']]))
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    p.add_argument('--asset'); p.add_argument('--ckpt'); p.add_argument('--cameras'); p.add_argument('--out', required=True)
    p.add_argument('--label', default=''); p.add_argument('--max-cameras', type=int, default=400)
    p.add_argument('--native-cache', default='', help='dir for compiled decoders (default <out dir>/native-cache)')
    a = p.parse_args()
    if bool(a.asset) == bool(a.ckpt): raise SystemExit('FAILED: give exactly one of --asset / --ckpt')
    xyz, faces, dc, mapping, residual, bank, rbank, meta = from_asset(a.asset, a.stage2, a.native_cache or Path(a.out).resolve().parent / 'native-cache') if a.asset else from_ckpt(a.ckpt)
    if faces.max() >= len(xyz): raise SystemExit('FAILED: face index out of range')
    if mapping.max() >= len(bank): raise SystemExit('FAILED: mapping out of range')
    if len(rbank) and residual[residual != 0xFFFFFFFF].size and residual[residual != 0xFFFFFFFF].max() >= len(rbank): raise SystemExit('FAILED: residual mapping out of range')
    blobs = [('xyz', xyz), ('faces', faces), ('dc', dc), ('mapping', mapping), ('residual', residual), ('bank', bank), ('residual_bank', rbank)]
    lo, hi = xyz.min(0).tolist(), xyz.max(0).tolist()
    header = dict(version=1, label=a.label or Path(a.out).stem, V=int(len(xyz)), F=int(len(faces)), K=int(len(bank)), R=int(len(rbank)),
                  bbox=[lo, hi], cameras=load_cameras(a.cameras, a.max_cameras), meta=meta,
                  blobs=[dict(name=n, dtype=str(b.dtype), shape=list(b.shape), bytes=int(b.nbytes)) for n, b in blobs])
    hb = json.dumps(header, separators=(',', ':')).encode(); hb += b' ' * (-len(hb) % 4)   # keep blobs 4-byte aligned for typed-array views
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        f.write(b'MSV1'); f.write(struct.pack('<I', len(hb))); f.write(hb)
        for _, b in blobs: f.write(np.ascontiguousarray(b).astype(b.dtype.newbyteorder('<') if b.dtype.byteorder == '>' else b.dtype).tobytes())
    print(f"VIEWER {header['label']} | V {header['V']} F {header['F']} K {header['K']} R {header['R']} | cameras {len(header['cameras'])} | "
          f"source {meta['source']} {meta['file_bytes']/1e6:.3f} MB | decode {meta.get('decode_read_s', '-')} s read, {meta['decode_total_s']} s total | "
          f"msv {out.stat().st_size/1e6:.1f} MB -> {out}")

if __name__ == '__main__': main()
PYEOF_V1
cat > $VW/viewer.html <<'HTMLEOF_V1'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>MeshSplat viewer v1</title>
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
    <button id="prev">◀ cam</button><button id="next">cam ▶</button>
    <label><input type="checkbox" id="spin"> spin</label>
    <button id="bench">benchmark (300 frames)</button>
  </div>
  <div class="row" id="stats"><span class="mut">no file loaded</span></div>
  <div class="row mut" id="info"></div>
  <div class="row" id="benchline" style="user-select:text;-webkit-user-select:text;word-break:break-all"></div>
  <div class="row mut" id="help">drag: orbit · wheel / pinch: dolly · two-finger / shift-drag: pan · W A S D Q E: move</div>
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
precision mediump float; in vec3 vColor; out vec4 o; void main() { o = vec4(vColor, 1.0); }`;
function compile(type, src) { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s)); return s; }
const prog = gl.createProgram(); gl.attachShader(prog, compile(gl.VERTEX_SHADER, vertexShaderSource())); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
const U = {}; for (const n of ['uViewProj', 'uCam', 'uBank', 'uRBank', 'uDeg']) U[n] = gl.getUniformLocation(prog, n);

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
let camIndex = 0, diag = 1;
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
  if (mesh) { gl.deleteVertexArray(mesh.vao); gl.deleteTexture(mesh.bank); gl.deleteTexture(mesh.rbank); }
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const attr = (loc, data, size, integer) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc); if (integer) gl.vertexAttribIPointer(loc, size, gl.UNSIGNED_INT, 0, 0); else gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0); };
  attr(0, blobs.xyz, 3, false); attr(1, blobs.dc, 3, false);
  const idx = new Uint32Array(header.V * 2); for (let i = 0; i < header.V; i++) { idx[2*i] = blobs.mapping[i]; idx[2*i+1] = blobs.residual[i]; }
  attr(2, idx, 2, true);
  const ebo = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, blobs.faces, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  const bank = bankTexture(blobs.bank, header.K); const rbank = bankTexture(header.R ? blobs.residual_bank : new Float32Array(45), Math.max(1, header.R));
  mesh = { header, vao, ebo, bank, rbank, V: header.V, F: header.F, name };
  const [lo, hi] = header.bbox; diag = v3.len(v3.sub(hi, lo));
  camIndex = 0; setCamera(0);
  const m = header.meta || {};
  setInfo(`${header.label} · V ${header.V.toLocaleString()} · F ${header.F.toLocaleString()} · K ${header.K} R ${header.R} · source ${m.source || '?'} ${(m.file_bytes/1e6).toFixed(2)} MB` +
          (m.decode_total_s != null ? ` · decode ${m.decode_total_s} s` : '') + ` · cameras ${header.cameras.length} · ${gpuName}`);
  $('drop').style.display = 'none';
}

function setCamera(i) {
  const cams = mesh.header.cameras;
  if (!cams.length) { // no cameras: look at bbox centre from +z
    const [lo, hi] = mesh.header.bbox; const ctr = v3.scale(v3.add(lo, hi), 0.5);
    cam = { pos: v3.add(ctr, [0, 0, -1.5 * diag]), R: [[1,0,0],[0,-1,0],[0,0,1]], fx: 800, fy: 800, w: 1200, h: 800, dist: 1.5 * diag, name: 'default' }; return;
  }
  i = ((i % cams.length) + cams.length) % cams.length; camIndex = i; const c = cams[i];
  // rotation is C2W: column j = camera axis j in world (x right, y down, z forward)
  const R = [[c.rotation[0][0], c.rotation[1][0], c.rotation[2][0]], [c.rotation[0][1], c.rotation[1][1], c.rotation[2][1]], [c.rotation[0][2], c.rotation[1][2], c.rotation[2][2]]];
  cam = { pos: c.position.slice(), R, fx: c.fx, fy: c.fy, w: c.width, h: c.height, dist: 0.3 * diag, name: c.name || String(c.id) };
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
  gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(0, 0, 0, 1); gl.enable(gl.DEPTH_TEST); gl.disable(gl.CULL_FACE);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!mesh) return;
  gl.useProgram(prog); gl.bindVertexArray(mesh.vao);
  gl.uniformMatrix4fv(U.uViewProj, false, viewProj()); gl.uniform3fv(U.uCam, cam.pos); gl.uniform1i(U.uDeg, parseInt($('deg').value));
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, mesh.bank); gl.uniform1i(U.uBank, 0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, mesh.rbank); gl.uniform1i(U.uRBank, 1);
  gl.drawElements(gl.TRIANGLES, mesh.F * 3, gl.UNSIGNED_INT, 0); gl.bindVertexArray(null);
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
function orbit(dyaw, dpitch) { // rotate camera about its target around the camera's own up / right axes
  const [r, d, fw] = cam.R; const up = v3.scale(d, -1); const target = v3.add(cam.pos, v3.scale(fw, cam.dist));
  let rel = v3.sub(cam.pos, target), R = cam.R;
  rel = rot(rel, up, -dyaw); R = R.map(a => rot(a, up, -dyaw));
  const right = R[0]; rel = rot(rel, right, -dpitch); R = R.map(a => rot(a, right, -dpitch));
  cam.pos = v3.add(target, rel); cam.R = R;
}
function dolly(f) { const fw = cam.R[2]; const step = cam.dist * f; cam.pos = v3.add(cam.pos, v3.scale(fw, step)); cam.dist = Math.max(0.01 * diag, cam.dist - step); }
function pan(dx, dy) { const [r, d] = cam.R; const s = cam.dist * 0.002; cam.pos = v3.add(cam.pos, v3.add(v3.scale(r, -dx * s), v3.scale(d, -dy * s))); }
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
window.addEventListener('keydown', e => { if (!cam) return; const s = 0.02 * diag; const [r, d, fw] = cam.R; const k = e.key.toLowerCase();
  const mv = { w: fw, s: v3.scale(fw, -1), a: v3.scale(r, -1), d: r, q: d, e: v3.scale(d, -1) }[k]; if (mv) cam.pos = v3.add(cam.pos, v3.scale(mv, s)); });
$('prev').onclick = () => mesh && setCamera(camIndex - 1); $('next').onclick = () => mesh && setCamera(camIndex + 1);
$('bench').onclick = benchmark; $('res').onchange = resize;
$('file').onchange = e => { const f = e.target.files[0]; if (!f) return; setStatus('reading ' + f.name + ' …'); f.arrayBuffer().then(b => loadMesh(b, f.name)).catch(err => { setStatus('FAILED: ' + err.message); console.error(err); }); };
const q = new URLSearchParams(location.search).get('file');
if (q) { setStatus('fetching ' + q + ' …'); fetch(q).then(async r => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); const total = +r.headers.get('content-length') || 0; const reader = r.body.getReader(); const chunks = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (total) $('bar').style.width = (100 * got / total) + '%'; }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; } $('bar').style.width = '0'; return buf.buffer; })
  .then(b => loadMesh(b, q)).catch(err => { setStatus('FAILED: ' + err.message); console.error(err); }); }
</script>
</body>
</html>
HTMLEOF_V1
python3 -c "import ast,sys; ast.parse(open('$VW/export_viewer_v1.py').read()); print('export_viewer_v1.py ok')" && grep -c "BENCH" $VW/viewer.html && ls -la $VW
```

## §2 export all scenes + decode times (GPU node; ~10-15 min total, CPU only)

```bash
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v1; PKG=$AN/vis_collapse-v12
cat > $VW/export-all-v1.sh <<'SHEOF_V1'
#!/usr/bin/env bash
# export-all-v1.sh: decode the delivered asset of every scene (lambda 6.2e-3 = the ~73x point) + Room lambda 1.85e-3
# into viewer .msv files; log decode times. Never exits the shell: prints FAILED lines and continues.
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v1; PKG=$AN/vis_collapse-v12; OUT=$VW/msv; mkdir -p $OUT
SCENES=${SCENES:-"room garden bonsai counter kitchen bicycle stump flowers treehill truck train"}
LOG=$VW/export-v1.log; : > $LOG
run_one() {  # scene lam
  local S=$1 L=$2 G=/scratch/zr2245/$1-rd3-lam$2-mu1.0 A CAM
  A=$G/axisB/v1.0/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $G/axisB/v1.0/deliver -name asset.mslb 2>/dev/null | head -1)
  CAM=$G/original/cameras.json; [ -f "$CAM" ] || CAM=/scratch/zr2245/$S-vis-collapse/original/cameras.json
  [ -f "$A" ] || { echo "FAILED: no asset.mslb under $G/axisB/v1.0/deliver" | tee -a $LOG; return; }
  [ -f "$CAM" ] || { echo "FAILED: no cameras.json for $S (looked in $G/original and $S-vis-collapse/original)" | tee -a $LOG; return; }
  echo "=== $S lam $L ($(date +%H:%M:%S)) ===" | tee -a $LOG
  bash $PKG/stage2_run.sh $VW/export_viewer_v1.py --asset $A --cameras $CAM --out $OUT/$S-lam$L.msv --label "$S lam$L" --native-cache $VW/native-cache > $VW/export-$S-lam$L.log 2>&1 \
    || { echo "FAILED: export $S lam $L"; tail -8 $VW/export-$S-lam$L.log; } | tee -a $LOG
  grep "^VIEWER" $VW/export-$S-lam$L.log | tee -a $LOG
}
for S in $SCENES; do run_one $S 0.0062; done
run_one room 0.00185
grep "^VIEWER" $LOG > $VW/decode-times-v1.txt
echo "===== $VW/decode-times-v1.txt ====="; cat $VW/decode-times-v1.txt; echo "----- failures -----"; grep "^FAILED" $LOG || echo none; du -sh $OUT
SHEOF_V1
bash $VW/export-all-v1.sh
```

Each `VIEWER` line = `scene | V F K R | cameras | source MB | decode X s read, Y s total | msv MB`. "read" = container + connectivity + XYZ + SH-index decode; "total" adds dequantisation. Paste `decode-times-v1.txt` to me.

Optional, desktop only (per-vertex SH, ~1.3 M codes -> ~250 MB GPU texture; phones will refuse it): the uncompressed Room original for a side-by-side FPS.
```bash
AN=/scratch/zr2245/analysis; VW=$AN/viewer-v1; PKG=$AN/vis_collapse-v12
bash $PKG/stage2_run.sh $VW/export_viewer_v1.py --ckpt /scratch/zr2245/room-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt --cameras /scratch/zr2245/room-vis-collapse/original/cameras.json --out $VW/msv/room-original.msv --label "room original" 2>&1 | grep -v "^stage2 package"
```

## §3 desktop FPS (your laptop)

1. Copy the viewer and a few .msv files down (sizes ~30-60 MB each; start with room / garden / bicycle):
   ```bash
   mkdir -p ~/msv && scp 'torch:/scratch/zr2245/analysis/viewer-v1/viewer.html' 'torch:/scratch/zr2245/analysis/viewer-v1/msv/room-lam0.0062.msv' 'torch:/scratch/zr2245/analysis/viewer-v1/msv/garden-lam0.0062.msv' 'torch:/scratch/zr2245/analysis/viewer-v1/msv/bicycle-lam0.0062.msv' ~/msv/
   ```
   (`torch` = whatever host alias torch_connect.sh uses; or download from the VS Code explorer.)
2. Open `~/msv/viewer.html` in Chrome (double-click; `file://` is fine), choose a .msv with the file button. Resolution stays "dataset resolution" (the training image size, so FPS is comparable across devices); SH degree 3.
3. Press **benchmark (300 frames)**; copy the `BENCH ...` line (button, or select the line under the info text). Repeat for each scene; paste the lines to me.
4. Also note the display FPS shown while orbiting (vsync-capped, so the BENCH number is the real one).

## §4 phone FPS (same Wi-Fi as the laptop)

```bash
cd ~/msv && python3 -m http.server 8000
```
Laptop IP: `ipconfig getifaddr en0` (Mac) / `hostname -I` (Linux). On the phone open
`http://<laptop-ip>:8000/viewer.html?file=room-lam0.0062.msv`
(iOS Safari 15+ / Android Chrome, WebGL2 required). Wait for the progress bar, press benchmark, long-press the BENCH line to copy, or screenshot it. Do room + garden + bicycle at least; the per-vertex original is desktop only.

## §5 what the numbers become

Table 2 row per scene: bytes | train min (walltime-v1) | decode s (decode-times-v1) | laptop fps | phone fps | "standard mesh: yes". Fig. 1 inset = phone screenshot of the viewer with the fps overlay. Ask me for the table once the three outputs (gap-table-v1.txt, walltime-v1.txt, BENCH lines) are in.
