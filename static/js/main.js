// === SelfEvo Website — Main Controller ===
// Side-by-side viewers: pretrained (left) vs SelfEvo (right).

const viewers = {};
const sectionInited = {};

function initSection(sectionKey, preCanvasId, preLoadingId, preGlfailedId,
                                  evoCanvasId, evoLoadingId, evoGlfailedId,
                                  scenesId, inputImgId) {
  // Prevent double init
  if (sectionInited[sectionKey]) return;
  sectionInited[sectionKey] = true;

  let preViewer = null, evoViewer = null;
  try {
    preViewer = createViewer({
      canvasId:     preCanvasId,
      loadingId:    preLoadingId,
      glfailedId:   preGlfailedId,
      sectionKey:   sectionKey,
      fixedVersion: 'pretrained',
    });
  } catch(e) { console.error('[main] pretrained viewer failed:', e); }
  try {
    evoViewer = createViewer({
      canvasId:     evoCanvasId,
      loadingId:    evoLoadingId,
      glfailedId:   evoGlfailedId,
      sectionKey:   sectionKey,
      fixedVersion: 'selfevo',
    });
  } catch(e) { console.error('[main] selfevo viewer failed:', e); }

  viewers[sectionKey + '-pre'] = preViewer;
  viewers[sectionKey + '-evo'] = evoViewer;

  // Sync camera controls between both viewers (delta-based so each keeps its own init)
  if (preViewer && evoViewer) {
    const CAM_KEYS = ['distance','forward','elevation','zoom','rx','ry','tx','ty'];
    function applySyncDelta(source, target, cam) {
      var srcBase = source.getBaseCamera();
      var tgtBase = target.getBaseCamera();
      var synced = {};
      CAM_KEYS.forEach(function(k) { synced[k] = tgtBase[k] + (cam[k] - srcBase[k]); });
      target.syncCamera(synced);
    }
    preViewer.onCameraChange(cam => applySyncDelta(preViewer, evoViewer, cam));
    evoViewer.onCameraChange(cam => applySyncDelta(evoViewer, preViewer, cam));

    preViewer.onPlaybackChange(playing => evoViewer.syncPlayback(playing));
    evoViewer.onPlaybackChange(playing => preViewer.syncPlayback(playing));
  }

  // --- Scene thumbnails ---
  const scenes = SCENE_CONFIG[sectionKey] || [];
  const scenesContainer = document.getElementById(scenesId);
  if (scenesContainer && scenes.length > 0) {
    scenes.forEach((scene, idx) => {
      const thumb = document.createElement('img');
      thumb.className = 'scene-thumb' + (idx === 0 ? ' active' : '');
      thumb.src = `static/images/thumbs/${scene.name}.jpg`;
      thumb.alt = scene.label;
      thumb.title = scene.label;
      thumb.dataset.scene = scene.name;
      if (scene.movie) thumb.dataset.movie = String(scene.movie);

      thumb.onerror = function() {
        const pill = document.createElement('button');
        pill.className = 'scene-pill' + (idx === 0 ? ' active' : '');
        pill.textContent = scene.label;
        pill.dataset.scene = scene.name;
        if (scene.movie) pill.dataset.movie = String(scene.movie);
        pill.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer, scene.camera, scene.state, scene.cameraPre));
        this.replaceWith(pill);
      };

      thumb.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer, scene.camera, scene.state, scene.cameraPre));
      scenesContainer.appendChild(thumb);
    });

    // Wire up input frame display (driven by evo viewer)
    const inputImgEl = inputImgId ? document.getElementById(inputImgId) : null;
    const driverViewer = evoViewer || preViewer;
    if (inputImgEl && driverViewer) {
      driverViewer.onFrameChange((frame) => {
        const url = driverViewer.getFrameUrl(frame);
        if (url) inputImgEl.src = url;
      });
    }

    // For adaptation, apply movie 1 filter by default; otherwise load first scene
    if (sectionKey === 'adaptation') {
      applyMovieFilter('adaptation', 1);
    } else {
      if (preViewer) preViewer.loadScene(scenes[0].name, scenes[0].cameraPre || scenes[0].camera, scenes[0].state);
      if (evoViewer) evoViewer.loadScene(scenes[0].name, scenes[0].camera, scenes[0].state);
      syncViewerButtons(sectionKey, scenes[0].state);

      // Preload next few scenes in the background
      var driver = preViewer || evoViewer;
      if (driver && scenes.length > 1) {
        [1, 2, 3].forEach(function(i) {
          if (i < scenes.length) driver.preloadSceneData(scenes[i].name);
        });
      }
    }
  }
}

function syncViewerButtons(sectionKey, stateOverride) {
  const s = stateOverride || {};
  const pointsBtn = document.getElementById(sectionKey === 'generalization' ? 'gen-btn-points' : 'adapt-btn-points');
  const frustaBtn = document.getElementById(sectionKey === 'generalization' ? 'gen-btn-frusta' : 'adapt-btn-frusta');
  if (pointsBtn) pointsBtn.classList.toggle('active', s.other_points === 'points');
  if (frustaBtn) frustaBtn.classList.toggle('active', !!s.other_frusta);
}

function selectScene(sectionKey, sceneName, container, cameraOverride, stateOverride, cameraPreOverride) {
  container.querySelectorAll('.scene-thumb, .scene-pill').forEach(el => {
    el.classList.toggle('active', el.dataset.scene === sceneName);
  });
  const pre = viewers[sectionKey + '-pre'];
  const evo = viewers[sectionKey + '-evo'];
  if (pre) pre.loadScene(sceneName, cameraPreOverride || cameraOverride, stateOverride);
  if (evo) evo.loadScene(sceneName, cameraOverride, stateOverride);
  syncViewerButtons(sectionKey, stateOverride);

  // Preload adjacent scenes so next click is near-instant
  var scenes = SCENE_CONFIG[sectionKey] || [];
  var idx = scenes.findIndex(function(s) { return s.name === sceneName; });
  if (idx >= 0) {
    var driver = pre || evo;
    [idx - 1, idx + 1, idx + 2].forEach(function(i) {
      if (i >= 0 && i < scenes.length && driver) {
        driver.preloadSceneData(scenes[i].name);
      }
    });
  }
}

function resetBothViewers(sectionKey) {
  const pre = viewers[sectionKey + '-pre'];
  const evo = viewers[sectionKey + '-evo'];
  if (pre) pre.resetCamera();
  if (evo) evo.resetCamera();
}

function toggleBothViewers(sectionKey, type, btnEl) {
  const pre = viewers[sectionKey + '-pre'];
  const evo = viewers[sectionKey + '-evo'];
  if (type === 'points') {
    if (pre) pre.togglePoints();
    if (evo) evo.togglePoints();
    const v = pre || evo;
    btnEl.classList.toggle('active', !!(v && v.state.other_points === 'points'));
  } else {
    if (pre) pre.toggleFrusta();
    if (evo) evo.toggleFrusta();
    const v = pre || evo;
    btnEl.classList.toggle('active', !!(v && v.state.other_frusta));
  }
}

function applyMovieFilter(sectionKey, movieNum) {
  const container = document.getElementById(sectionKey === 'adaptation' ? 'adapt-scenes' : null);
  if (!container) return;
  const scenes = SCENE_CONFIG[sectionKey] || [];

  // Show/hide elements
  container.querySelectorAll('.scene-thumb, .scene-pill').forEach(el => {
    el.style.display = el.dataset.movie === String(movieNum) ? '' : 'none';
  });

  // Find and load the first scene of this movie
  const firstScene = scenes.find(s => s.movie === movieNum);
  if (firstScene) {
    container.querySelectorAll('.scene-thumb, .scene-pill').forEach(el => {
      el.classList.toggle('active', el.dataset.scene === firstScene.name);
    });
    const pre = viewers[sectionKey + '-pre'];
    const evo = viewers[sectionKey + '-evo'];
    if (pre) pre.loadScene(firstScene.name, firstScene.camera, firstScene.state);
    if (evo) evo.loadScene(firstScene.name, firstScene.camera, firstScene.state);
    syncViewerButtons(sectionKey, firstScene.state);
  }
}

function selectAdaptMovie(movieNum, btnEl) {
  document.querySelectorAll('#adapt-movie-pagination .movie-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.movie === String(movieNum));
  });
  applyMovieFilter('adaptation', movieNum);
}

// --- Pause/resume viewers based on section visibility ---
function setupVisibilityManager(sectionId, sectionKey) {
  var section = document.getElementById(sectionId);
  if (!section) return;
  new IntersectionObserver(function(entries) {
    var visible = entries[0].isIntersecting;
    var pre = viewers[sectionKey + '-pre'];
    var evo = viewers[sectionKey + '-evo'];
    if (visible) {
      if (pre) pre.resume();
      if (evo) evo.resume();
    } else {
      if (pre) pre.pause();
      if (evo) evo.pause();
    }
  }, { threshold: 0 }).observe(section);
}

// --- Lazy-init sections when they scroll near the viewport ---
function setupLazyInit(sectionId, initFn) {
  var section = document.getElementById(sectionId);
  if (!section) return;
  var obs = new IntersectionObserver(function(entries) {
    if (entries[0].isIntersecting) {
      obs.disconnect();
      initFn();
    }
  }, { rootMargin: '200px 0px' }); // init 200px before entering viewport
  obs.observe(section);
}

// --- Initialize on page load ---
document.addEventListener('DOMContentLoaded', () => {
  // Generalization is near the top — init immediately (don't wait for full page load)
  initSection('generalization',
    'gen-pre-canvas', 'gen-pre-loading', 'gen-pre-glfailed',
    'gen-evo-canvas', 'gen-evo-loading', 'gen-evo-glfailed',
    'gen-scenes', 'gen-input-frame');

  // Adaptation is further down — lazy-init when near viewport
  setupLazyInit('adaptation', function() {
    initSection('adaptation',
      'adapt-pre-canvas', 'adapt-pre-loading', 'adapt-pre-glfailed',
      'adapt-evo-canvas', 'adapt-evo-loading', 'adapt-evo-glfailed',
      'adapt-scenes', 'adapt-input-frame');
    setupVisibilityManager('adaptation', 'adaptation');
  });

  // Pause/resume generalization viewers based on visibility
  setupVisibilityManager('generalization', 'generalization');
});
