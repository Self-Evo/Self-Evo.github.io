// === SelfEvo Viewer — Instance-based WebGL Point Cloud Viewer ===
// Adapted from MegaSaM's viewer.js to support multiple viewers and
// pretrained/selfevo toggle with camera state preservation.

// ============================================================
//  SHADERS
// ============================================================

const vertexShaders = {
xyz_rgba: `#version 300 es
  precision highp float;
  in vec3 xyz;
  in vec4 rgba;
  uniform mat4 camera;
  uniform float point_size;
  uniform float minWidth;
  out vec4 color;
  in float index;

  void main(void) {
    gl_Position = camera * vec4(xyz, 1.0);
    float size = point_size / gl_Position.w;
    color = rgba;
    if (size < minWidth) {
      color.a *= size / minWidth;
      size = minWidth;
    }
    gl_PointSize = size;
  }`,

xy: `#version 300 es
  precision highp float;
  in vec2 xy;
  uniform mat4 camera;
  uniform mat4 pose;
  uniform float frustum_size;
  out vec2 v_uv;

  void main(void) {
    gl_Position = camera * pose * (vec4(frustum_size, frustum_size, frustum_size, 1.0) * vec4(xy, 1.0, 1.0));
    v_uv = xy;
  }`,

depth: `#version 300 es
  precision highp float;
  uniform mat4 camera;
  uniform mat4 pose;
  uniform sampler2D depth;
  uniform float depthscale;
  uniform float point_size;
  uniform int width;
  uniform int height;
  uniform int stride;
  uniform float max_grad;
  out vec2 v_uv;

  float d(vec2 p) {
    vec4 rgba = texture(depth, p);
    return depthscale * (rgba.r + rgba.g/256.0);
  }

  void main(void) {
    int x = gl_VertexID % (width / stride) * stride;
    int y = gl_VertexID / (width / stride) * stride;
    vec2 uv;
    uv.x = (float(x) + 0.5) / float(width);
    uv.y = (float(y) + 0.5) / float(height);
    highp float z = d(uv);
    vec2 dx = vec2(1.0 / float(width), 0.0);
    vec2 dy = vec2(0.0, 1.0 / float(height));
    highp float gx = abs(d(uv + dx) - d(uv - dx));
    highp float gy = abs(d(uv + dy) - d(uv - dy));
    if (gx > max_grad * z || gy > max_grad * z) {
      z = 0.0;
    }
    gl_Position = camera * pose * vec4(uv.x * z, uv.y * z, z, 1.0);
    v_uv = uv;
    gl_PointSize = point_size * z / gl_Position[3];
  }`,

linesegment: `#version 300 es
  precision highp float;
  uniform mat4 camera;
  uniform float width;
  uniform float height;
  uniform float lineWidth;
  uniform float minWidth;
  in vec3 xyz0;
  in vec3 xyz1;
  in vec4 rgba;
  out vec4 color;

  void main(void) {
    vec4 zero4 = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 p0 = camera * vec4(xyz0, 1.0);
    vec4 p1 = camera * vec4(xyz1, 1.0);
    color = rgba;
    if (p0.w < 0.0 || p1.w < 0.0) {
      gl_Position = zero4;
      color = zero4;
      return;
    }
    float p0w = p0.w;
    float p1w = p1.w;
    p0 /= p0w;
    p1 /= p1w;
    float r0 = lineWidth / p0w;
    float r1 = lineWidth / p1w;
    float r0a = 1.0;
    float r1a = 1.0;
    if (r0 < minWidth) { r0a = r0 / minWidth; r0 = minWidth; }
    if (r1 < minWidth) { r1a = r1 / minWidth; r1 = minWidth; }

    vec2 viewsize = vec2(width, height);
    vec2 unit = (p1.xy - p0.xy) * viewsize;
    float linelength = length(unit);
    unit /= linelength;
    float theta = asin((r0 - r1) / linelength);
    vec4 p;
    float r;
    float side = float(2*(gl_VertexID % 2) - 1);
    if (gl_VertexID < 2) {
      p = p0; r = r0; color.a *= r0a;
    } else {
      p = p1; r = r1; color.a *= r1a;
    }
    vec2 offset = vec2(-unit.y, unit.x);
    gl_Position = p + vec4(
      (unit * (sin(theta) * r) + offset * cos(theta) * side * r) / viewsize,
      0.0, 0.0);
  }`,
};

const fragmentShaders = {
vcolor: `#version 300 es
  precision highp float;
  in vec4 color;
  out vec4 outColor;
  void main(void) { outColor = color; }`,

tex: `#version 300 es
  precision highp float;
  in highp vec2 v_uv;
  uniform sampler2D image;
  uniform float alpha;
  out vec4 color;
  void main(void) {
    color = texture(image, v_uv);
    color.a *= alpha;
  }`,

roundpoint: `#version 300 es
  precision highp float;
  in vec4 color;
  out vec4 outColor;
  void main(void) {
    vec2 d = 2.0*gl_PointCoord - vec2(1.0, 1.0);
    if (dot(d, d) > 1.0) discard;
    outColor = color;
  }`
};

const programDefs = {
  screen: ['xy', 'tex'],
  cloud: ['depth', 'tex'],
  linequads: ['linesegment', 'vcolor'],
  roundpoints: ['xyz_rgba', 'roundpoint'],
};

// ============================================================
//  STATIC BUFFERS
// ============================================================

const FRUSTUM_DATA = [0,0,0, 0,0,1, 0,0,0, 1,0,1, 0,0,0, 0,1,1, 0,0,0, 1,1,1,
                      0,0,1, 1,0,1, 1,0,1, 1,1,1, 1,1,1, 0,1,1, 0,1,1, 0,0,1];
const FRUSTUM_POINTS_DATA = [0,0,0, 0,0,1, 0,1,1, 1,0,1, 1,1,1];
const CORNERS_DATA = [0,0, 1,0, 0,1, 1,1];

// ============================================================
//  MATRIX UTILITIES
// ============================================================

function matCompose(var_args) {
  const l = arguments.length;
  if (l === 0) return matI();
  let m = arguments[0];
  for (let i = 1; i < l; i++) m = matMM(m, arguments[i]);
  return m;
}

function matMM(a, b) {
  const c = [];
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      const k = j * 4;
      c.push(a[k]*b[i] + a[k+1]*b[i+4] + a[k+2]*b[i+8] + a[k+3]*b[i+12]);
    }
  }
  return c;
}

function vec4Lerp(a, b, p) {
  const q = 1 - p;
  return [a[0]*q+b[0]*p, a[1]*q+b[1]*p, a[2]*q+b[2]*p, a[3]*q+b[3]*p];
}

function matRx(t) {
  const c = Math.cos(t), s = Math.sin(t);
  return [1,0,0,0, 0,c,-s,0, 0,s,c,0, 0,0,0,1];
}
function matRy(t) {
  const c = Math.cos(t), s = Math.sin(t);
  return [c,0,s,0, 0,1,0,0, -s,0,c,0, 0,0,0,1];
}
function matScale(t) {
  return [t,0,0,0, 0,t,0,0, 0,0,t,0, 0,0,0,1];
}
function matI() {
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
}
function matT([x, y, z]) {
  return [1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1];
}

function rgba(text) {
  function f(a, b) {
    const x = parseInt(a, 16);
    const y = b ? parseInt(b, 16) : x;
    return (isNaN(x) || isNaN(y)) ? 128 : 16*x + y;
  }
  if (text[0] !== '#') text = '#' + text;
  let col = [255,0,255,255];
  switch (text.length) {
    case 4: col = [f(text[1]), f(text[2]), f(text[3]), 255]; break;
    case 7: col = [f(text[1],text[2]), f(text[3],text[4]), f(text[5],text[6]), 255]; break;
  }
  return [col[0]/255, col[1]/255, col[2]/255, col[3]/255];
}

// ============================================================
//  .packed FETCH
// ============================================================

async function fetchPacked(url) {
  const results = {};
  const response = await fetch(url);
  if (response.status !== 200) {
    console.error('fetchPacked error', response.status, await response.text());
    return results;
  }
  const blob = await response.blob();
  const prefix_size = new DataView(await blob.slice(0, 8).arrayBuffer()).getUint32(0, true);
  const json = JSON.parse(await blob.slice(8, prefix_size).text());
  for (const [key, [start, end, content_type]] of Object.entries(json)) {
    results[key] = blob.slice(start + prefix_size, end + prefix_size, content_type);
  }
  return results;
}

function pad5(i) { return i.toString().padStart(5, '0'); }

// ============================================================
//  DEFAULT PARAMETERS
// ============================================================

const defaultCamera = {
  distance: 0.4,
  forward: 0,
  elevation: 0.1,
  zoom: 1,
  follow: false,
  follow_rotation: true,
  rx: 0.3,
  ry: 0,
};

const defaultState = {
  draw_frustum: true,
  image: 'image',
  show_points: 'points',
  points_alpha: 1.0,
  stride: 1,
  point_size: 4,
  frustum_size: 0.25,
  z_clamp: 0.02,
  show_back_facing: false,
  every_nth: 10,
  reveal: false,
  other_offset: 0,
  other_frusta: false,
  other_image: 'empty',
  other_points: 'no',
  other_points_alpha: 1.0,
  other_stride: 2,
  other_point_size: 1,
  playing: true,
  fps: 10,
  frame: 0,
  camera_frame: 0,
  background: 0.95,
  frustum_width: 0.8,
  other_frusta_width: 0.4,
  draw_path: false,
  path_width: 2,
  path_dash: 0,
};

const defaultColors = {
  frustum: '#ff0000',
  other_frusta: '#6666ff',
  other_frusta_end: '#6666ff',
  path: '#888888',
};

function cloneDefaults(spec) {
  return Object.assign({}, spec);
}

// ============================================================
//  VIEWER CLASS
// ============================================================

function createViewer(config) {
  // config: { canvasId, loadingId, glfailedId, sectionKey, fixedVersion? }
  const canvasEl = document.getElementById(config.canvasId);
  const loadingEl = document.getElementById(config.loadingId);
  const glfailedEl = document.getElementById(config.glfailedId);

  if (!canvasEl) {
    console.error('[viewer] canvas not found:', config.canvasId);
    return null;
  }

  let gl;
  try {
    gl = canvasEl.getContext('webgl2', {antialias: false, alpha: false, preserveDrawingBuffer: true});
  } catch(e) {
    console.error('[viewer] WebGL2 context error:', e);
  }
  if (!gl) {
    if (glfailedEl) { glfailedEl.style.display = 'flex'; glfailedEl.style.alignItems = 'center'; glfailedEl.style.justifyContent = 'center'; }
    return null;
  }

  // Per-instance state
  const camera = cloneDefaults(defaultCamera);
  const state = cloneDefaults(defaultState);
  const colors = cloneDefaults(defaultColors);
  const cameraInternal = { near: 0.01, far: 100, aspect_ratio: 1, xfrac: 1 };

  let dirty = true;
  let remaining_to_load = 0;
  let total_to_load = 0;
  let frame_time = Date.now();
  let loop_pause_until = 0;
  let lastNotifiedFrame = -1;
  let frameChangeCallback = null;

  // Dual data: pretrained and selfevo
  const sceneData = { pretrained: null, selfevo: null };
  let currentVersion = config.fixedVersion || 'selfevo';
  let currentSceneName = null;

  // Active data reference
  function getData() { return sceneData[currentVersion]; }

  // --- GL setup ---
  const programs = {};
  for (const k in programDefs) programs[k] = [...programDefs[k]];

  const buffers = {
    frustum: [...FRUSTUM_DATA],
    frustum_points: [...FRUSTUM_POINTS_DATA],
    corners: [...CORNERS_DATA],
  };

  function initShadersGL(shaders, type) {
    const compiled = {};
    for (const i in shaders) {
      const s = gl.createShader(type);
      gl.shaderSource(s, shaders[i]);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.log('Compiling ' + i, gl.getShaderInfoLog(s));
      }
      compiled[i] = s;
    }
    return compiled;
  }

  function initProgramsGL(vs, fs, progs) {
    for (const i in progs) {
      const p = gl.createProgram();
      gl.attachShader(p, vs[progs[i][0]]);
      gl.attachShader(p, fs[progs[i][1]]);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.log('Linking ' + i, gl.getProgramInfoLog(p));
      }
      const attribute = {};
      const uniform = {};
      let n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES);
      for (let j = 0; j < n; ++j) {
        const info = gl.getActiveAttrib(p, j);
        const loc = gl.getAttribLocation(p, info.name);
        if (loc >= 0) attribute[info.name] = loc;
      }
      n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let j = 0; j < n; ++j) {
        const info = gl.getActiveUniform(p, j);
        const loc = gl.getUniformLocation(p, info.name);
        if (loc) uniform[info.name] = loc;
      }
      progs[i].name = i;
      progs[i].program = p;
      progs[i].attribute = attribute;
      progs[i].uniform = uniform;
    }
  }

  function initBuffer(b) {
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    const data = new Float32Array(b);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    b.vb = vb;
    b.data = data;
  }

  function initBuffersGL(bufs) {
    for (const b in bufs) initBuffer(bufs[b]);
  }

  // --- Loading ---
  function updateLoading() {
    if (remaining_to_load <= 0 && loadingEl) {
      loadingEl.style.display = 'none';
    }
  }

  function showLoading() {
    if (loadingEl) loadingEl.style.display = 'flex';
  }

  function loadTexture(blob) {
    const t = gl.createTexture();
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = function() {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      t.ready = true;
      URL.revokeObjectURL(url);
      dirty = true;
      remaining_to_load--;
      updateLoading();
    };
    total_to_load++;
    remaining_to_load++;
    image.src = url;
    return t;
  }

  function loadDepth(blob) {
    const t = gl.createTexture();
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = function() {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      t.ready = true;
      URL.revokeObjectURL(url);
      dirty = true;
      remaining_to_load--;
      updateLoading();
    };
    total_to_load++;
    remaining_to_load++;
    image.src = url;
    return t;
  }

  function buildPathBuffer(poses) {
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    const floats = new Float32Array(poses.length * 3);
    let idx = 0;
    for (const p of poses) {
      floats[idx] = p[0][3]; floats[idx+1] = p[1][3]; floats[idx+2] = p[2][3];
      idx += 3;
    }
    gl.bufferData(gl.ARRAY_BUFFER, floats, gl.STATIC_DRAW);
    return {vb, n: poses.length};
  }

  function parsePackedData(packed_data) {
    const d = JSON.parse(packed_data['data.json'] instanceof Blob
      ? null // handled by async
      : packed_data['data.json']);
    return d;
  }

  async function loadVersionData(packed_data) {
    const textContent = await packed_data['data.json'].text();
    const d = JSON.parse(textContent);
    d.rgb = [];
    d.depth = [];
    d.rgbBlobs = [];   // raw blobs for input frame display
    d.rgbUrls  = [];   // lazy object URL cache
    d.depth_scale = d.depth_scale || 20;
    d.path = buildPathBuffer(d.poses);

    for (let i = 0; i < d.poses.length; i++) {
      const blob = packed_data[`rgb_${pad5(i+1)}.png`];
      d.rgbBlobs.push(blob);
      d.rgbUrls.push(null);
      d.rgb[i] = loadTexture(blob);
      d.depth[i] = loadDepth(packed_data[`depthrgb_${pad5(i+1)}.png`]);
    }
    return d;
  }

  // --- Scene Loading ---
  async function loadScene(sceneName, cameraOverride) {
    if (currentSceneName === sceneName) return;
    currentSceneName = sceneName;
    showLoading();
    remaining_to_load = 0;
    total_to_load = 0;

    // Reset state but keep camera for smooth scene switching
    state.frame = 0;
    state.playing = true;
    loop_pause_until = 0;

    // Load only needed versions (fixedVersion loads one; otherwise load both)
    const loadPretrained = !config.fixedVersion || config.fixedVersion === 'pretrained';
    const loadSelfevo    = !config.fixedVersion || config.fixedVersion === 'selfevo';

    const prePromise = loadPretrained
      ? fetchPacked(`static/packed/pretrained/${sceneName}.packed`).catch(() => null)
      : Promise.resolve(null);
    const sePromise = loadSelfevo
      ? fetchPacked(`static/packed/selfevo/${sceneName}.packed`).catch(() => null)
      : Promise.resolve(null);

    const [prePacked, sePacked] = await Promise.all([prePromise, sePromise]);

    if (prePacked && prePacked['data.json']) {
      sceneData.pretrained = await loadVersionData(prePacked);
    } else {
      sceneData.pretrained = null;
    }

    if (sePacked && sePacked['data.json']) {
      sceneData.selfevo = await loadVersionData(sePacked);
    } else {
      sceneData.selfevo = null;
    }

    // Reset camera for new scene, applying per-scene overrides if provided
    Object.assign(camera, cloneDefaults(defaultCamera));
    if (cameraOverride) Object.assign(camera, cameraOverride);
    dirty = true;
  }

  // --- Toggle version (preserving camera) ---
  function setVersion(version) {
    if (version === currentVersion) return;
    currentVersion = version;
    // Preserve frame index, clamp if needed
    const data = getData();
    if (data && data.poses) {
      state.frame = Math.min(state.frame, data.poses.length - 1);
    }
    dirty = true;
  }

  // --- Camera matrix ---
  function cameraMatrix(cam, pose) {
    const d = 1 / (cameraInternal.far - cameraInternal.near);
    const a = (cameraInternal.near + cameraInternal.far) * d;
    const b = -2 * (cameraInternal.near * cameraInternal.far) * d;
    const w = cam.zoom;
    const h = cam.zoom * cameraInternal.aspect_ratio;
    const px = cameraInternal.xfrac - 1;
    const perspective = [w,0,px,0, 0,-h,0,0, 0,0,a,b, 0,0,1,0];
    const target = [0, 0, -cam.forward];
    const follow_position = matT([-pose[0][3], -pose[1][3], -pose[2][3]]);
    let follow_rotation;
    if (cam.follow_rotation) {
      follow_rotation = [
        pose[0][0], pose[1][0], pose[2][0], 0,
        pose[0][1], pose[1][1], pose[2][1], 0,
        pose[0][2], pose[1][2], pose[2][2], 0,
        0, 0, 0, 1];
    } else {
      follow_rotation = matI();
    }
    return matCompose(
      perspective,
      matT([0, cam.elevation * cam.distance, cam.distance]),
      matRx(cam.rx), matRy(cam.ry),
      matT(target),
      follow_rotation,
      follow_position);
  }

  function poseMatrix(data, i) {
    const intrinsics = data.intrinsics[i];
    const px = intrinsics[2], py = intrinsics[3];
    const ifx = 1.0 / intrinsics[0], ify = 1.0 / intrinsics[1];
    const tex_to_cam = [ifx,0,-px*ifx,0, 0,ify,-py*ify,0, 0,0,1,0, 0,0,0,1];
    const cp = data.poses[i];
    const pose = [
      cp[0][0],cp[0][1],cp[0][2],cp[0][3],
      cp[1][0],cp[1][1],cp[1][2],cp[1][3],
      cp[2][0],cp[2][1],cp[2][2],cp[2][3],
      0,0,0,1];
    return matCompose(pose, tex_to_cam);
  }

  // --- Drawing ---
  function useProgram(p) {
    gl.useProgram(p.program);
    for (const a of Object.values(p.attribute)) {
      gl.disableVertexAttribArray(a);
      gl.vertexAttribDivisor(a, 0);
    }
  }

  function* otherFrames(data) {
    const step = state.every_nth || 1;
    const limit = state.reveal ? state.frame : data.poses.length;
    for (let i = state.other_offset; i < limit; i += step) {
      let fade = 1.0;
      if (state.reveal && i > state.frame - step) fade = (state.frame - i) / step;
      yield [i, fade];
    }
  }

  function totalPoints(data, stride) {
    return (data.width / stride | 0) * (data.height / stride | 0);
  }

  function prepareDrawFrustum(camera_matrix, size_factor) {
    const p = programs['linequads'];
    useProgram(p);
    gl.uniform1f(p.uniform.width, canvasEl.width);
    gl.uniform1f(p.uniform.height, canvasEl.height);
    gl.uniform1f(p.uniform.minWidth, size_factor);
    if (p.uniform.zOffset) gl.uniform1f(p.uniform.zOffset, 0.0);
    if (p.uniform.maxLength) gl.uniform1i(p.uniform.maxLength, 1);
    if (p.uniform.maxSolidLength) gl.uniform1i(p.uniform.maxSolidLength, 1);
    if (p.uniform.recolor) gl.uniform1i(p.uniform.recolor, 0);
    if (p.uniform.index_stride) gl.uniform1i(p.uniform.index_stride, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.frustum.vb);
    gl.enableVertexAttribArray(p.attribute.xyz0);
    gl.enableVertexAttribArray(p.attribute.xyz1);
    gl.vertexAttribDivisor(p.attribute.xyz0, 1);
    gl.vertexAttribDivisor(p.attribute.xyz1, 1);
    if (p.attribute.segmentLength !== undefined) gl.vertexAttrib1f(p.attribute.segmentLength, 0.0);
    if (p.attribute.index !== undefined) gl.vertexAttrib1f(p.attribute.index, 0.0);
    gl.vertexAttribPointer(p.attribute.xyz0, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribPointer(p.attribute.xyz1, 3, gl.FLOAT, false, 24, 12);
  }

  function drawFrustum(camera_matrix, data, i, color, width) {
    const p = programs['linequads'];
    gl.uniformMatrix4fv(p.uniform.camera, true, matCompose(
        camera_matrix, poseMatrix(data, i), matScale(state.frustum_size)));
    gl.vertexAttrib4fv(p.attribute.rgba, color);
    gl.uniform1f(p.uniform.lineWidth, width);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, 8);
  }

  function prepareDrawFrustumPoints(camera_matrix, size_factor) {
    const p = programs['roundpoints'];
    useProgram(p);
    gl.uniform1f(p.uniform.width, canvasEl.width);
    gl.uniform1f(p.uniform.height, canvasEl.height);
    gl.uniform1f(p.uniform.minWidth, size_factor);
    if (p.uniform.zOffset) gl.uniform1f(p.uniform.zOffset, 0.0);
    if (p.uniform.recolor) gl.uniform1i(p.uniform.recolor, 0);
    if (p.uniform.index_stride) gl.uniform1i(p.uniform.index_stride, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.frustum_points.vb);
    gl.enableVertexAttribArray(p.attribute.xyz);
    if (p.attribute.index !== undefined) gl.vertexAttrib1f(p.attribute.index, 0.0);
    gl.vertexAttribPointer(p.attribute.xyz, 3, gl.FLOAT, false, 12, 0);
  }

  function drawFrustumPoints(camera_matrix, data, i, color, size) {
    const p = programs['roundpoints'];
    gl.uniformMatrix4fv(p.uniform.camera, true, matCompose(
        camera_matrix, poseMatrix(data, i), matScale(state.frustum_size)));
    gl.vertexAttrib4fv(p.attribute.rgba, color);
    gl.uniform1f(p.uniform.point_size, size);
    gl.drawArrays(gl.POINTS, 0, 5);
  }

  function drawPath(camera_matrix, data, color, size_factor, width, dash) {
    const p = programs['linequads'];
    gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
    gl.vertexAttrib4fv(p.attribute.rgba, color);
    gl.uniform1f(p.uniform.minWidth, size_factor);
    if (p.uniform.zOffset) gl.uniform1f(p.uniform.zOffset, 0.0);
    gl.uniform1f(p.uniform.lineWidth, width);
    gl.bindBuffer(gl.ARRAY_BUFFER, data.path.vb);
    const step = (2*dash) || 1;
    gl.vertexAttribPointer(p.attribute.xyz0, 3, gl.FLOAT, false, step * 12, 0);
    gl.vertexAttribPointer(p.attribute.xyz1, 3, gl.FLOAT, false, step * 12, (dash || 1) * 12);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, Math.floor((data.path.n-1) / step));
  }

  function prepareDrawImage(camera_matrix) {
    const p = programs['screen'];
    useProgram(p);
    gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.corners.vb);
    gl.enableVertexAttribArray(p.attribute.xy);
    gl.vertexAttribPointer(p.attribute.xy, 2, gl.FLOAT, false, 8, 0);
    gl.uniform1f(p.uniform.frustum_size, state.frustum_size * 1.001);
  }

  function drawImage(data, frame, tex, alpha) {
    if (!tex.ready) return;
    const p = programs['screen'];
    gl.uniformMatrix4fv(p.uniform.pose, true, poseMatrix(data, frame));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(p.uniform.image, 0);
    gl.uniform1f(p.uniform.alpha, alpha);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, buffers.corners.length / 2);
  }

  function prepareDrawPoints(camera_matrix, data, stride, size, size_factor) {
    const p = programs['cloud'];
    useProgram(p);
    gl.uniformMatrix4fv(p.uniform.camera, true, camera_matrix);
    gl.uniform1i(p.uniform.image, 0);
    gl.uniform1i(p.uniform.depth, 1);
    gl.uniform1f(p.uniform.depthscale, data.depth_scale);
    gl.uniform1f(p.uniform.max_grad, state.z_clamp * 2);
    gl.uniform1f(p.uniform.point_size, size * size_factor);
    gl.uniform1i(p.uniform.width, data.width);
    gl.uniform1i(p.uniform.height, data.height);
    gl.uniform1i(p.uniform.stride, stride);
  }

  function drawPoints(data, i, stride, alpha) {
    const depth_tex = data.depth[i];
    if (!data.rgb[i].ready || !depth_tex.ready) return;
    const p = programs['cloud'];
    gl.uniform1f(p.uniform.alpha, alpha);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, data.rgb[i]);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, depth_tex);
    gl.uniformMatrix4fv(p.uniform.pose, true, poseMatrix(data, i));
    gl.drawArrays(gl.POINTS, 0, totalPoints(data, stride));
  }

  function drawSceneContent(camera_matrix, data, size_factor, opaque) {
    if (!state.show_back_facing) {
      gl.enable(gl.CULL_FACE);
      gl.cullFace(gl.FRONT);
    }

    // Other points
    if (state.other_points === 'points') {
      prepareDrawPoints(camera_matrix, data, state.other_stride, state.other_point_size, size_factor);
      for (const [i, fade] of otherFrames(data)) {
        if ((fade === 1.0) === opaque) {
          drawPoints(data, i, state.other_stride, fade * state.other_points_alpha);
        }
      }
    }

    if (opaque) {
      if (state.show_points === 'points') {
        prepareDrawPoints(camera_matrix, data, state.stride, state.point_size, size_factor);
        drawPoints(data, state.frame, state.stride, state.points_alpha);
      }
    }
    gl.disable(gl.CULL_FACE);
  }

  function draw() {
    const data = getData();
    const back = state.background;
    gl.clearColor(back, back, back, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!data || !data.poses) return;

    const size_factor = state.size_factor;
    const camera_matrix = cameraMatrix(camera, data.poses[state.camera_frame]);

    gl.disable(gl.CULL_FACE);

    // Opaque pass
    drawSceneContent(camera_matrix, data, size_factor, true);

    // Frusta lines
    prepareDrawFrustum(camera_matrix, size_factor);
    if (state.other_frusta) {
      const c0 = rgba(colors.other_frusta);
      const c1 = rgba(colors.other_frusta_end);
      for (const [i, fade] of otherFrames(data)) {
        if (i !== state.frame || !state.draw_frustum) {
          drawFrustum(camera_matrix, data, i,
            vec4Lerp(c0, c1, i / (data.poses.length - 1)),
            state.other_frusta_width * size_factor);
        }
      }
    }
    if (state.draw_frustum) {
      drawFrustum(camera_matrix, data, state.frame, rgba(colors.frustum), state.frustum_width * size_factor);
    }
    if (state.draw_path) {
      drawPath(camera_matrix, data, rgba(colors.path), size_factor, state.path_width * size_factor, state.path_dash);
    }

    // Frustum points
    prepareDrawFrustumPoints(camera_matrix, size_factor);
    if (state.other_frusta) {
      const c0 = rgba(colors.other_frusta);
      const c1 = rgba(colors.other_frusta_end);
      for (const [i, fade] of otherFrames(data)) {
        if (i !== state.frame || !state.draw_frustum) {
          drawFrustumPoints(camera_matrix, data, i,
            vec4Lerp(c0, c1, i / (data.poses.length - 1)),
            state.other_frusta_width * size_factor);
        }
      }
    }
    if (state.draw_frustum) {
      drawFrustumPoints(camera_matrix, data, state.frame, rgba(colors.frustum), state.frustum_width * size_factor);
    }

    // Images in frusta
    prepareDrawImage(camera_matrix);
    if (state.other_image !== 'empty') {
      for (const [i, fade] of otherFrames(data)) {
        if (i !== state.frame || state.image === 'empty') {
          drawImage(data, i, data.rgb[i], 1.0);
        }
      }
    }
    if (state.image !== 'empty') {
      drawImage(data, state.frame, data.rgb[state.frame], 1.0);
    }

    // Non-opaque pass
    drawSceneContent(camera_matrix, data, size_factor, false);
  }

  // --- Animation ---
  function update() {
    const time = Date.now();
    // Hold on the last frame briefly before looping back to frame 0
    if (loop_pause_until > 0) {
      if (time < loop_pause_until) return;
      loop_pause_until = 0;
      const data = getData();
      if (state.playing && data && data.poses && data.poses.length) {
        state.frame = 0;
        frame_time = time;
        dirty = true;
      }
      return;
    }
    const period = 1000.0 / state.fps;
    if (time - frame_time <= period) return;
    frame_time = time;
    const data = getData();
    if (state.playing && data && data.poses && data.poses.length) {
      const next = (state.frame + 1) % data.poses.length;
      if (next === 0) {
        loop_pause_until = time + 350; // 350 ms pause at loop boundary
        return;
      }
      state.frame = next;
      dirty = true;
    }
  }

  function tick() {
    window.requestAnimationFrame(tick);
    if (remaining_to_load > 0) return;
    if (state.playing) update();
    if (state.frame !== lastNotifiedFrame) {
      lastNotifiedFrame = state.frame;
      if (frameChangeCallback) frameChangeCallback(state.frame);
    }
    if (dirty) {
      dirty = false;
      const data = getData();
      if (data) {
        state.camera_frame = camera.follow ? state.frame : 0;
      }
      draw();
    }
  }

  // --- Input handlers ---
  function addHandlers() {
    let dragging = false;
    let ox, oy, rxStart, ryStart;
    const speed = Math.PI / 500;

    canvasEl.addEventListener('pointerdown', (e) => {
      dragging = true;
      ox = e.clientX; oy = e.clientY;
      rxStart = camera.rx; ryStart = camera.ry;
      canvasEl.setPointerCapture(e.pointerId);
    });
    canvasEl.addEventListener('pointermove', (e) => {
      if (dragging) {
        camera.rx = rxStart + (e.clientY - oy) * speed;
        camera.ry = ryStart - (e.clientX - ox) * speed;
        dirty = true;
      }
    });
    const endDrag = () => { dragging = false; };
    canvasEl.addEventListener('pointerup', endDrag);
    canvasEl.addEventListener('pointercancel', endDrag);
    canvasEl.addEventListener('pointerout', endDrag);
    canvasEl.addEventListener('pointerleave', endDrag);

    canvasEl.addEventListener('wheel', (e) => {
      const delta = e.deltaY || -e.wheelDeltaY;
      camera.distance = Math.max(0, Math.min(10, camera.distance + delta * 0.001));
      dirty = true;
      e.preventDefault();
      e.stopPropagation();
    });

    canvasEl.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (handleKey(e.key, e.shiftKey)) {
        dirty = true;
        e.preventDefault();
        e.stopPropagation();
      }
    });

    // Make canvas focusable for keyboard events
    canvasEl.tabIndex = 0;
  }

  function handleKey(key, shift) {
    const data = getData();
    if (!data || !data.poses) return false;
    switch (key) {
      case 'ArrowLeft':
        state.playing = false;
        state.frame = (state.frame - (shift ? 10 : 1) + data.poses.length) % data.poses.length;
        return true;
      case 'ArrowRight':
        state.playing = false;
        state.frame = (state.frame + (shift ? 10 : 1)) % data.poses.length;
        return true;
      case ' ':
        state.playing = !state.playing;
        return true;
    }
    return false;
  }

  // --- Resize ---
  function resize() {
    const dp = window.devicePixelRatio;
    state.size_factor = dp;
    const w = canvasEl.clientWidth * dp;
    const h = canvasEl.clientHeight * dp;
    canvasEl.width = w;
    canvasEl.height = h;
    gl.viewport(0, 0, w, h);
    cameraInternal.aspect_ratio = w / h;
    dirty = true;
  }

  // --- Public API ---
  function togglePoints() {
    if (state.other_points === 'no') {
      state.other_points = 'points';
      state.every_nth = 1;
    } else {
      state.other_points = 'no';
      if (!state.other_frusta) state.every_nth = defaultState.every_nth;
    }
    dirty = true;
  }

  function toggleFrusta() {
    if (!state.other_frusta) {
      state.other_frusta = true;
      state.every_nth = 1;
    } else {
      state.other_frusta = false;
      if (state.other_points === 'no') state.every_nth = defaultState.every_nth;
    }
    dirty = true;
  }

  // --- Initialize ---
  resize();
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const v = initShadersGL(vertexShaders, gl.VERTEX_SHADER);
  const f = initShadersGL(fragmentShaders, gl.FRAGMENT_SHADER);
  initProgramsGL(v, f, programs);
  initBuffersGL(buffers);
  addHandlers();
  window.addEventListener('resize', resize);
  window.requestAnimationFrame(tick);

  function onFrameChange(cb) { frameChangeCallback = cb; }

  function getFrameUrl(i) {
    // Use selfevo data if available, else pretrained (input frames are the same)
    const data = sceneData.selfevo || sceneData.pretrained;
    if (!data || !data.rgbBlobs) return null;
    const idx = (i !== undefined) ? i : state.frame;
    if (idx < 0 || idx >= data.rgbBlobs.length) return null;
    if (!data.rgbUrls[idx]) {
      data.rgbUrls[idx] = URL.createObjectURL(data.rgbBlobs[idx]);
    }
    return data.rgbUrls[idx];
  }

  return {
    loadScene,
    setVersion,
    togglePoints,
    toggleFrusta,
    onFrameChange,
    getFrameUrl,
    get currentVersion() { return currentVersion; },
    get currentScene() { return currentSceneName; },
    get state() { return state; },
    get camera() { return camera; },
  };
}
