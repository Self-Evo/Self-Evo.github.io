// === SelfEvo Website — Main Controller ===
// Side-by-side viewers: pretrained (left) vs SelfEvo (right).

const viewers = {};

function initSection(sectionKey, preCanvasId, preLoadingId, preGlfailedId,
                                  evoCanvasId, evoLoadingId, evoGlfailedId,
                                  scenesId) {
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

      thumb.onerror = function() {
        const pill = document.createElement('button');
        pill.className = 'scene-pill' + (idx === 0 ? ' active' : '');
        pill.textContent = scene.label;
        pill.dataset.scene = scene.name;
        pill.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer));
        this.replaceWith(pill);
      };

      thumb.addEventListener('click', () => selectScene(sectionKey, scene.name, scenesContainer));
      scenesContainer.appendChild(thumb);
    });

    // Load first scene in both viewers
    if (scenes.length > 0) {
      if (preViewer) preViewer.loadScene(scenes[0].name);
      if (evoViewer) evoViewer.loadScene(scenes[0].name);
    }
  }
}

function selectScene(sectionKey, sceneName, container) {
  container.querySelectorAll('.scene-thumb, .scene-pill').forEach(el => {
    el.classList.toggle('active', el.dataset.scene === sceneName);
  });
  const pre = viewers[sectionKey + '-pre'];
  const evo = viewers[sectionKey + '-evo'];
  if (pre) pre.loadScene(sceneName);
  if (evo) evo.loadScene(sceneName);
}

// --- Initialize on page load ---
window.addEventListener('load', () => {
  initSection('adaptation',
    'adapt-pre-canvas', 'adapt-pre-loading', 'adapt-pre-glfailed',
    'adapt-evo-canvas', 'adapt-evo-loading', 'adapt-evo-glfailed',
    'adapt-scenes');
  initSection('generalization',
    'gen-pre-canvas', 'gen-pre-loading', 'gen-pre-glfailed',
    'gen-evo-canvas', 'gen-evo-loading', 'gen-evo-glfailed',
    'gen-scenes');
});
