// === SelfEvo Website — Main Controller ===
// Side-by-side viewers: pretrained (left) vs SelfEvo (right).

const viewers = {};

function initSection(sectionKey, preCanvasId, preLoadingId, preGlfailedId,
                                  evoCanvasId, evoLoadingId, evoGlfailedId,
                                  scenesId, inputImgId) {
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
        pill.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer, scene.camera, scene.state));
        this.replaceWith(pill);
      };

      thumb.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer, scene.camera, scene.state));
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
      if (preViewer) preViewer.loadScene(scenes[0].name, scenes[0].camera, scenes[0].state);
      if (evoViewer) evoViewer.loadScene(scenes[0].name, scenes[0].camera, scenes[0].state);
      syncViewerButtons(sectionKey, scenes[0].state);
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

function selectScene(sectionKey, sceneName, container, cameraOverride, stateOverride) {
  container.querySelectorAll('.scene-thumb, .scene-pill').forEach(el => {
    el.classList.toggle('active', el.dataset.scene === sceneName);
  });
  const pre = viewers[sectionKey + '-pre'];
  const evo = viewers[sectionKey + '-evo'];
  if (pre) pre.loadScene(sceneName, cameraOverride, stateOverride);
  if (evo) evo.loadScene(sceneName, cameraOverride, stateOverride);
  syncViewerButtons(sectionKey, stateOverride);
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

// --- Initialize on page load ---
window.addEventListener('load', () => {
  initSection('adaptation',
    'adapt-pre-canvas', 'adapt-pre-loading', 'adapt-pre-glfailed',
    'adapt-evo-canvas', 'adapt-evo-loading', 'adapt-evo-glfailed',
    'adapt-scenes', 'adapt-input-frame');
  initSection('generalization',
    'gen-pre-canvas', 'gen-pre-loading', 'gen-pre-glfailed',
    'gen-evo-canvas', 'gen-evo-loading', 'gen-evo-glfailed',
    'gen-scenes', 'gen-input-frame');
});
