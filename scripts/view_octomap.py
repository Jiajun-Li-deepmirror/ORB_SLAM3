import argparse
import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import octomap

TEMPLATE = r"""<title>{title}</title>
<style>
  :root {{
    /* Deliberately single-theme (dark viewport): a point-cloud/occupancy viewer only
       reads correctly against a dark ground, same convention as CloudCompare/RViz/Blender. */
    --bg: #0b0e13;
    --surface: #141a22;
    --surface-2: #1b222c;
    --border: #262e3a;
    --text: #e8ecf1;
    --text-dim: #8b93a3;
    --accent: #4fd1c5;
    --accent-dim: #2e7d76;
  }}

  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: "IBM Plex Sans", system-ui, sans-serif;
  }}
  canvas {{ display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}

  .hud {{
    position: fixed;
    top: 16px;
    left: 16px;
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    max-width: 300px;
    backdrop-filter: blur(6px);
  }}
  .hud h1 {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 10px;
  }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 12px;
    line-height: 1.7;
  }}
  .stat-label {{ color: var(--text-dim); }}
  .stat-value {{ font-variant-numeric: tabular-nums; color: var(--text); }}
  .hint {{
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 11.5px;
    color: var(--text-dim);
    line-height: 1.6;
  }}
  .hint kbd {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 10.5px;
    color: var(--text);
  }}

  .legend {{
    position: fixed;
    bottom: 16px;
    left: 16px;
    background: color-mix(in srgb, var(--surface) 88%, transparent);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    backdrop-filter: blur(6px);
  }}
  .legend-title {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 6px;
  }}
  .legend-bar {{
    width: 180px;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3730a3, #2563eb, #06b6d4, #22c55e, #eab308, #f97316, #ef4444);
  }}
  .legend-labels {{
    display: flex;
    justify-content: space-between;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 3px;
  }}

  #loading {{
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 13px;
    color: var(--text-dim);
    letter-spacing: 0.05em;
  }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">

<div id="loading">DECODING POINT CLOUD&hellip;</div>

<div class="hud" style="display:none" id="hud">
  <h1>{title}</h1>
  <div class="stat-row"><span class="stat-label">points</span><span class="stat-value" id="stat-points">&mdash;</span></div>
  <div class="stat-row"><span class="stat-label">extent x</span><span class="stat-value" id="stat-x">&mdash;</span></div>
  <div class="stat-row"><span class="stat-label">extent y</span><span class="stat-value" id="stat-y">&mdash;</span></div>
  <div class="stat-row"><span class="stat-label">extent z</span><span class="stat-value" id="stat-z">&mdash;</span></div>
  <div class="hint">
    <kbd>drag</kbd> rotate &nbsp; <kbd>scroll</kbd> zoom &nbsp; <kbd>shift+drag</kbd> pan
  </div>
</div>

<div class="legend" style="display:none" id="legend">
  <div class="legend-title">Height (Z)</div>
  <div class="legend-bar"></div>
  <div class="legend-labels"><span id="legend-min">&mdash;</span><span id="legend-max">&mdash;</span></div>
</div>

<canvas id="gl"></canvas>

<script>
const POINTS_B64 = "{points_b64}";

function decodePoints(b64) {{
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}

const flat = decodePoints(POINTS_B64);
const numPoints = flat.length / 3;

let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
for (let i = 0; i < numPoints; i++) {{
  const x = flat[i*3], y = flat[i*3+1], z = flat[i*3+2];
  if (x < xmin) xmin = x; if (x > xmax) xmax = x;
  if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  if (z < zmin) zmin = z; if (z > zmax) zmax = z;
}}
const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2, cz = (zmin + zmax) / 2;
const extent = Math.max(xmax - xmin, ymax - ymin, zmax - zmin) || 1;

function turbo(t) {{
  const stops = [
    [0.216,0.192,0.643],[0.145,0.388,0.882],[0.059,0.702,0.812],
    [0.204,0.729,0.341],[0.902,0.808,0.129],[0.949,0.482,0.145],[0.902,0.204,0.204]
  ];
  const n = stops.length - 1;
  const f = Math.min(Math.max(t, 0), 1) * n;
  const i = Math.min(Math.floor(f), n - 1);
  const frac = f - i;
  const a = stops[i], b = stops[i+1];
  return [a[0]+(b[0]-a[0])*frac, a[1]+(b[1]-a[1])*frac, a[2]+(b[2]-a[2])*frac];
}}

const verts = new Float32Array(numPoints * 6);
for (let i = 0; i < numPoints; i++) {{
  const x = flat[i*3] - cx, y = flat[i*3+1] - cy, z = flat[i*3+2] - cz;
  verts[i*6+0] = x; verts[i*6+1] = y; verts[i*6+2] = z;
  const t = (flat[i*3+2] - zmin) / Math.max(zmax - zmin, 1e-6);
  const [r,g,bl] = turbo(t);
  verts[i*6+3] = r; verts[i*6+4] = g; verts[i*6+5] = bl;
}}

document.getElementById('loading').style.display = 'none';
document.getElementById('hud').style.display = 'block';
document.getElementById('legend').style.display = 'block';
document.getElementById('stat-points').textContent = numPoints.toLocaleString();
document.getElementById('stat-x').textContent = (xmax-xmin).toFixed(1) + ' m';
document.getElementById('stat-y').textContent = (ymax-ymin).toFixed(1) + ' m';
document.getElementById('stat-z').textContent = (zmax-zmin).toFixed(1) + ' m';
document.getElementById('legend-min').textContent = zmin.toFixed(1) + 'm';
document.getElementById('legend-max').textContent = zmax.toFixed(1) + 'm';

const PATH_B64 = {path_b64_json};

const canvas = document.getElementById('gl');
const gl = canvas.getContext('webgl', {{ antialias: true }});

function resize() {{
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = window.innerWidth * dpr;
  canvas.height = window.innerHeight * dpr;
  gl.viewport(0, 0, canvas.width, canvas.height);
}}
window.addEventListener('resize', resize);
resize();

const vsSource = `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uMatrix;
  uniform float uPointSize;
  varying vec3 vColor;
  void main() {{
    gl_Position = uMatrix * vec4(aPosition, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
  }}
`;
const fsSource = `
  precision mediump float;
  uniform bool uIsPoint;
  varying vec3 vColor;
  void main() {{
    if (uIsPoint) {{
      // gl_PointCoord is only meaningful for gl.POINTS - for gl.LINE_STRIP draws it's
      // undefined per spec, and several implementations leave it at (0,0), which makes
      // this circular-sprite mask discard EVERY line fragment (dot((-0.5,-0.5))=0.5 > 0.25)
      // and silently makes the whole line invisible. Skip the mask entirely for lines.
      vec2 c = gl_PointCoord - vec2(0.5);
      if (dot(c, c) > 0.25) discard;
    }}
    gl_FragColor = vec4(vColor, 1.0);
  }}
`;

function compile(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {{
    console.error(gl.getShaderInfoLog(s));
  }}
  return s;
}}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSource));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSource));
gl.linkProgram(prog);
gl.useProgram(prog);

const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);

const stride = 6 * 4;
const aPosition = gl.getAttribLocation(prog, 'aPosition');
gl.enableVertexAttribArray(aPosition);
gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, stride, 0);
const aColor = gl.getAttribLocation(prog, 'aColor');
gl.enableVertexAttribArray(aColor);
gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, stride, 3 * 4);

const uMatrix = gl.getUniformLocation(prog, 'uMatrix');
const uPointSize = gl.getUniformLocation(prog, 'uPointSize');
const uIsPoint = gl.getUniformLocation(prog, 'uIsPoint');

function bindAttribs(buffer) {{
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.vertexAttribPointer(aPosition, 3, gl.FLOAT, false, stride, 0);
  gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, stride, 3 * 4);
}}

const SPEED_B64 = {speed_b64_json};
const SPEED_MAX = {speed_max_json};

let pathBuf = null, pathCount = 0, markerBuf = null;
if (PATH_B64) {{
  const pathFlat = decodePoints(PATH_B64);
  const n = pathFlat.length / 3;
  const speedArr = SPEED_B64 ? decodePoints(SPEED_B64) : null;
  const pv = new Float32Array(n * 6);
  for (let i = 0; i < n; i++) {{
    pv[i*6+0] = pathFlat[i*3] - cx;
    pv[i*6+1] = pathFlat[i*3+1] - cy;
    pv[i*6+2] = pathFlat[i*3+2] - cz;
    if (speedArr) {{
      const t = SPEED_MAX > 0 ? Math.min(Math.max(speedArr[i] / SPEED_MAX, 0), 1) : 0;
      const [r,g,bl] = turbo(t);
      pv[i*6+3] = r; pv[i*6+4] = g; pv[i*6+5] = bl; // colored by speed
    }} else {{
      pv[i*6+3] = 1.0; pv[i*6+4] = 0.85; pv[i*6+5] = 0.2; // amber flight path (no speed data)
    }}
  }}
  pathBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, pathBuf);
  gl.bufferData(gl.ARRAY_BUFFER, pv, gl.STATIC_DRAW);
  pathCount = n;

  const mv = new Float32Array(2 * 6);
  mv[0]=pv[0]; mv[1]=pv[1]; mv[2]=pv[2]; mv[3]=0.25; mv[4]=0.92; mv[5]=0.45; // start, green
  const last = (n - 1) * 6;
  mv[6]=pv[last]; mv[7]=pv[last+1]; mv[8]=pv[last+2]; mv[9]=0.95; mv[10]=0.28; mv[11]=0.28; // goal, red
  markerBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, markerBuf);
  gl.bufferData(gl.ARRAY_BUFFER, mv, gl.STATIC_DRAW);

  const pathLegendLabel = speedArr
    ? '<span style="color:#3730a3;">&mdash;</span> trajectory (color = speed, 0-' + SPEED_MAX.toFixed(1) + ' m/s)'
    : '<span style="color:#ffd94d;">&mdash;</span> flight path';
  document.getElementById('legend').insertAdjacentHTML('beforeend',
    '<div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-family:\'IBM Plex Mono\',monospace;font-size:10.5px;color:var(--text-dim);">' +
    pathLegendLabel + ' &nbsp; ' +
    '<span style="color:#40eb73;">&#9679;</span> start &nbsp; ' +
    '<span style="color:#f24747;">&#9679;</span> goal</div>'
  );
}}

gl.enable(gl.DEPTH_TEST);
gl.clearColor(0x0b/255, 0x0e/255, 0x13/255, 1.0);

function mat4Multiply(a, b) {{
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {{
    for (let j = 0; j < 4; j++) {{
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k*4+j] * b[i*4+k];
      out[i*4+j] = sum;
    }}
  }}
  return out;
}}
function perspective(fovy, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const out = new Float32Array(16);
  out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf;
  out[11] = -1; out[14] = 2 * far * near * nf;
  return out;
}}
function translate(tx, ty, tz) {{
  const out = new Float32Array(16);
  out[0]=1; out[5]=1; out[10]=1; out[15]=1;
  out[12]=tx; out[13]=ty; out[14]=tz;
  return out;
}}
function rotateX(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  const out = new Float32Array(16);
  out[0]=1; out[5]=c; out[6]=s; out[9]=-s; out[10]=c; out[15]=1;
  return out;
}}
function rotateY(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  const out = new Float32Array(16);
  out[0]=c; out[2]=-s; out[5]=1; out[8]=s; out[10]=c; out[15]=1;
  return out;
}}

let yaw = 0.6, pitch = -0.5, dist = extent * 1.6;
let panX = 0, panY = 0;
let dragging = false, panning = false, lastX = 0, lastY = 0;

canvas.addEventListener('pointerdown', (e) => {{
  dragging = true; panning = e.shiftKey;
  lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener('pointerup', () => {{ dragging = false; }});
canvas.addEventListener('pointermove', (e) => {{
  if (!dragging) return;
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  if (panning) {{
    panX -= dx * dist * 0.0015;
    panY += dy * dist * 0.0015;
  }} else {{
    yaw += dx * 0.007;
    pitch += dy * 0.007;
    pitch = Math.max(-1.5, Math.min(1.5, pitch));
  }}
}});
canvas.addEventListener('wheel', (e) => {{
  e.preventDefault();
  dist *= Math.exp(e.deltaY * 0.001);
  dist = Math.max(extent * 0.05, Math.min(extent * 8, dist));
}}, {{ passive: false }});

function render() {{
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const aspect = canvas.width / canvas.height;
  const proj = perspective(Math.PI / 4, aspect, 0.01, extent * 20);
  const view = mat4Multiply(
    translate(panX, panY, -dist),
    mat4Multiply(rotateX(pitch), rotateY(yaw))
  );
  const mvp = mat4Multiply(proj, view);

  gl.uniformMatrix4fv(uMatrix, false, mvp);

  bindAttribs(buf);
  gl.uniform1i(uIsPoint, 1);
  gl.uniform1f(uPointSize, Math.max(1.5, 3.5 * (extent / dist)));
  gl.drawArrays(gl.POINTS, 0, numPoints);

  if (pathBuf) {{
    bindAttribs(pathBuf);
    gl.uniform1i(uIsPoint, 0);
    gl.lineWidth(1.0);
    gl.drawArrays(gl.LINE_STRIP, 0, pathCount);

    bindAttribs(markerBuf);
    gl.uniform1i(uIsPoint, 1);
    gl.uniform1f(uPointSize, Math.max(6.0, 14.0 * (extent / dist)));
    gl.drawArrays(gl.POINTS, 0, 2);
  }}

  requestAnimationFrame(render);
}}
render();
</script>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Render an Octomap .bt/.ot file as a self-contained interactive WebGL "
        "HTML page (drag to rotate, scroll to zoom, shift+drag to pan) - for viewing on a "
        "headless box with no octovis/RViz/display available. Publish the output with the "
        "Artifact tool, or open it directly in any browser."
    )
    parser.add_argument("octomap_path", type=str, help=".bt or .ot octomap file")
    parser.add_argument("out_html", type=str)
    parser.add_argument("--max_points", type=int, default=150000, help="randomly downsample occupied voxels above this count, for smooth interactive rendering")
    parser.add_argument("--title", type=str, default=None, help="defaults to the input filename's stem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--path_npz", type=str, default=None,
        help="overlay a planned path from a plan_path_3d.py output .npz (its 'path_world' "
        "array) - drawn as a line with green/red start/goal markers.",
    )
    args = parser.parse_args()

    tree = octomap.OcTree(str(args.octomap_path).encode())
    occupied, _empty = tree.extractPointCloud()
    print(f"Loaded {args.octomap_path}: {len(occupied)} occupied voxels (resolution={tree.getResolution()}m)")

    rng = np.random.default_rng(args.seed)
    if len(occupied) > args.max_points:
        idx = rng.choice(len(occupied), args.max_points, replace=False)
        occupied = occupied[idx]
    print(f"Using {len(occupied)} points after downsampling")

    path_b64_json = "null"
    speed_b64_json = "null"
    speed_max_json = "0"
    if args.path_npz:
        npz = np.load(args.path_npz)
        # Prefer the smoothed/speed-profiled trajectory (from the updated plan_path.py /
        # plan_path_3d.py) over the raw A* voxel path, when both are present in the npz.
        if "trajectory_xyz" in npz and "trajectory_v" in npz:
            path_world = npz["trajectory_xyz"].astype(np.float32)
            speed = npz["trajectory_v"].astype(np.float32)
            speed_b64_json = f'"{base64.b64encode(speed.tobytes()).decode("ascii")}"'
            speed_max_json = f"{float(speed.max()):.4f}" if len(speed) else "0"
            print(f"Overlaying trajectory: {len(path_world)} samples, peak speed {speed.max():.2f} m/s, from {args.path_npz}")
        else:
            path_world = npz["path_world"].astype(np.float32)
            print(f"Overlaying path: {len(path_world)} waypoints from {args.path_npz}")
        path_b64 = base64.b64encode(path_world.tobytes()).decode("ascii")
        path_b64_json = f'"{path_b64}"'

    b64 = base64.b64encode(occupied.astype(np.float32).tobytes()).decode("ascii")
    title = args.title or Path(args.octomap_path).stem
    html = TEMPLATE.format(
        title=title, points_b64=b64, path_b64_json=path_b64_json,
        speed_b64_json=speed_b64_json, speed_max_json=speed_max_json,
    )

    Path(args.out_html).write_text(html)
    print(f"Saved {args.out_html} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
