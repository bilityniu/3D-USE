(() => {
  "use strict";

  const scenes = {
    iui3: { label: "IUI3-RedSea", folder: "iui3" },
    curasao: { label: "Curasao", folder: "curasao" },
    japanese: { label: "Japanese Gardens", folder: "japanese" },
    panama: { label: "Panama", folder: "panama" },
    d3: { label: "D3", folder: "d3" },
  };

  const methods = {
    stage1: { label: "Stage 1", file: "stage1.mp4" },
    "3dgs": { label: "3DGS", file: "3dgs.mp4" },
    watersplatting: { label: "WaterSplatting", file: "watersplatting.mp4" },
    seasplat: { label: "SeaSplat", file: "seasplat.mp4" },
    marinestd: { label: "MarineSTD-GS", file: "marinestd-gs.mp4" },
    plenodium: { label: "Plenodium", file: "plenodium.mp4" },
    "3duir": { label: "3D-UIR", file: "3d-uir.mp4" },
  };
  const sceneOrder = Object.keys(scenes);

  document.querySelectorAll("[data-asset-frame]").forEach((frame) => {
    const image = frame.querySelector("[data-asset-image]");
    const placeholder = frame.querySelector("[data-asset-placeholder]");
    if (!image || !placeholder) return;

    const showImage = () => {
      frame.classList.add("has-asset");
      image.removeAttribute("hidden");
      placeholder.setAttribute("hidden", "");
    };
    const showPlaceholder = () => {
      frame.classList.remove("has-asset");
      image.setAttribute("hidden", "");
      placeholder.removeAttribute("hidden");
    };

    image.addEventListener("load", showImage);
    image.addEventListener("error", showPlaceholder);
    if (image.complete) image.naturalWidth > 0 ? showImage() : showPlaceholder();
  });

  function setVideoSource(video, source, placeholder) {
    placeholder.hidden = false;
    video.oncanplay = () => {
      placeholder.hidden = true;
    };
    video.onerror = () => {
      placeholder.hidden = false;
    };
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.src = source;
    video.load();
  }

  function createSynchronizedPlayback(videoList, master, playButton, restartButton) {
    let playing = false;
    let animationFrame = 0;

    const commonDuration = () => {
      const durations = videoList
        .map((video) => video.duration)
        .filter((duration) => Number.isFinite(duration) && duration > 0);
      return durations.length === videoList.length ? Math.min(...durations) : 0;
    };

    const seekAll = (time) => {
      videoList.forEach((video) => {
        if (video.readyState >= 1) video.currentTime = time;
      });
    };

    const updateButton = () => {
      playButton.textContent = playing ? "Pause" : "Play";
    };

    const synchronize = () => {
      if (!playing) return;
      const duration = commonDuration();
      if (duration && master.currentTime >= duration - 0.06) {
        seekAll(0);
        Promise.allSettled(videoList.map((video) => video.play()));
      } else {
        videoList.forEach((video) => {
          if (video !== master && Math.abs(video.currentTime - master.currentTime) > 0.1) {
            video.currentTime = master.currentTime;
          }
        });
      }
      animationFrame = requestAnimationFrame(synchronize);
    };

    const pause = () => {
      videoList.forEach((video) => video.pause());
      playing = false;
      cancelAnimationFrame(animationFrame);
      updateButton();
    };

    const play = async () => {
      const duration = commonDuration();
      if (duration && master.currentTime >= duration - 0.06) seekAll(0);
      seekAll(master.currentTime);
      await Promise.allSettled(videoList.map((video) => video.play()));
      playing = !master.paused;
      updateButton();
      cancelAnimationFrame(animationFrame);
      if (playing) synchronize();
    };

    const restart = () => {
      seekAll(0);
      if (playing) Promise.allSettled(videoList.map((video) => video.play()));
    };

    playButton.addEventListener("click", () => playing ? pause() : play());
    restartButton.addEventListener("click", restart);
    master.addEventListener("seeking", () => {
      videoList.forEach((video) => {
        if (video !== master && video.readyState >= 1) video.currentTime = master.currentTime;
      });
    });
    videoList.forEach((video) => video.addEventListener("ended", () => {
      if (playing) restart();
    }));

    return { pause, play, restart, isPlaying: () => playing };
  }

  function initEnhancementComparison() {
    const root = document.querySelector("#enhancement-comparison");
    if (!root) return;

    const videos = {
      left: root.querySelector('[data-enh-video="left"]'),
      right: root.querySelector('[data-enh-video="right"]'),
    };
    const placeholders = {
      left: root.querySelector('[data-enh-placeholder="left"]'),
      right: root.querySelector('[data-enh-placeholder="right"]'),
    };
    const playButton = root.querySelector("[data-enh-play]");
    const restartButton = root.querySelector("[data-enh-restart]");
    const state = { scene: "iui3", method: "stage1" };
    const playback = createSynchronizedPlayback(
      Object.values(videos), videos.left, playButton, restartButton
    );

    function updateLabels() {
      root.querySelectorAll("[data-enh-scene-label]").forEach((element) => {
        element.textContent = scenes[state.scene].label;
      });
      root.querySelectorAll("[data-enh-baseline-label]").forEach((element) => {
        element.textContent = methods[state.method].label;
      });
    }

    function loadSelection() {
      const folder = `media/enhancement/${scenes[state.scene].folder}`;
      const leftSource = `${folder}/${methods[state.method].file}`;
      playback.pause();
      setVideoSource(videos.left, leftSource, placeholders.left);
      setVideoSource(videos.right, `${folder}/3d-use.mp4`, placeholders.right);
      updateLabels();
    }

    function selectScene(sceneKey) {
      if (!scenes[sceneKey]) return;
      state.scene = sceneKey;
      root.querySelectorAll("[data-enh-scene]").forEach((button) => {
        button.classList.toggle("active", button.dataset.enhScene === sceneKey);
      });
      root.querySelectorAll("[data-enh-scene-dot]").forEach((button) => {
        const active = button.dataset.enhSceneDot === sceneKey;
        button.classList.toggle("active", active);
        active ? button.setAttribute("aria-current", "true") : button.removeAttribute("aria-current");
      });
      loadSelection();
    }

    function stepScene(direction) {
      const current = sceneOrder.indexOf(state.scene);
      selectScene(sceneOrder[(current + direction + sceneOrder.length) % sceneOrder.length]);
    }

    root.querySelectorAll("[data-enh-scene]").forEach((button) => {
      button.addEventListener("click", () => selectScene(button.dataset.enhScene));
    });
    root.querySelectorAll("[data-enh-scene-dot]").forEach((button) => {
      button.addEventListener("click", () => selectScene(button.dataset.enhSceneDot));
    });
    root.querySelector("[data-enh-scene-prev]").addEventListener("click", () => stepScene(-1));
    root.querySelector("[data-enh-scene-next]").addEventListener("click", () => stepScene(1));

    root.querySelectorAll("[data-enh-method]").forEach((button) => {
      button.addEventListener("click", () => {
        state.method = button.dataset.enhMethod;
        root.querySelectorAll("[data-enh-method]").forEach((item) => item.classList.toggle("active", item === button));
        loadSelection();
      });
    });

    loadSelection();
  }

  function initReconstructionComparison() {
    const root = document.querySelector("#reconstruction-comparison");
    if (!root) return;

    const stage = root.querySelector("[data-comparison-stage]");
    const overlay = root.querySelector("[data-comparison-overlay]");
    const divider = root.querySelector("[data-divider]");
    const lens = root.querySelector("[data-detail-lens]");
    const detailGrid = root.querySelector("[data-detail-grid]");
    const detailToggle = root.querySelector("[data-detail-toggle]");
    const lensSize = root.querySelector("[data-lens-size]");
    const sceneLabel = root.querySelector("[data-scene-label]");
    const playButton = root.querySelector("[data-recon-play]");
    const restartButton = root.querySelector("[data-recon-restart]");

    const videos = {
      left: root.querySelector('[data-recon-video="left"]'),
      right: root.querySelector('[data-recon-video="right"]'),
      leftZoom: root.querySelector('[data-recon-video="left-zoom"]'),
      rightZoom: root.querySelector('[data-recon-video="right-zoom"]'),
    };
    const placeholders = {
      left: root.querySelector('[data-placeholder="left"]'),
      right: root.querySelector('[data-placeholder="right"]'),
      leftZoom: root.querySelector('[data-crop-placeholder="left"]'),
      rightZoom: root.querySelector('[data-crop-placeholder="right"]'),
    };

    const state = {
      scene: "iui3",
      method: "3dgs",
      split: 50,
      dragging: null,
      lens: { x: 0.58, y: 0.42, size: 0.24 },
    };
    const playback = createSynchronizedPlayback(
      Object.values(videos), videos.left, playButton, restartButton
    );

    const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

    function updateLabels() {
      sceneLabel.textContent = scenes[state.scene].label;
      root.querySelectorAll("[data-baseline-label]").forEach((element) => {
        element.textContent = methods[state.method].label;
      });
    }

    function loadSelection() {
      const folder = `media/reconstruction/${scenes[state.scene].folder}`;
      const leftFile = methods[state.method].file;
      const leftSource = `${folder}/${leftFile}`;
      const rightSource = `${folder}/3d-use.mp4`;
      playback.pause();
      setVideoSource(videos.left, leftSource, placeholders.left);
      setVideoSource(videos.leftZoom, leftSource, placeholders.leftZoom);
      setVideoSource(videos.right, rightSource, placeholders.right);
      setVideoSource(videos.rightZoom, rightSource, placeholders.rightZoom);
      updateLabels();
    }

    function selectScene(sceneKey) {
      if (!scenes[sceneKey]) return;
      state.scene = sceneKey;
      root.querySelectorAll("[data-scene]").forEach((button) => {
        button.classList.toggle("active", button.dataset.scene === sceneKey);
      });
      root.querySelectorAll("[data-scene-dot]").forEach((button) => {
        const active = button.dataset.sceneDot === sceneKey;
        button.classList.toggle("active", active);
        active ? button.setAttribute("aria-current", "true") : button.removeAttribute("aria-current");
      });
      loadSelection();
    }

    function stepScene(direction) {
      const current = sceneOrder.indexOf(state.scene);
      selectScene(sceneOrder[(current + direction + sceneOrder.length) % sceneOrder.length]);
    }

    function updateSplit(clientX) {
      const rect = stage.getBoundingClientRect();
      state.split = clamp(((clientX - rect.left) / rect.width) * 100, 0, 100);
      overlay.style.clipPath = `inset(0 ${100 - state.split}% 0 0)`;
      divider.style.left = `${state.split}%`;
      divider.setAttribute("aria-valuenow", String(Math.round(state.split)));
    }

    function renderLens() {
      const { x, y, size } = state.lens;
      lens.style.left = `${x * 100}%`;
      lens.style.top = `${y * 100}%`;
      lens.style.width = `${size * 100}%`;
      lens.style.height = `${size * 100}%`;
      [videos.leftZoom, videos.rightZoom].forEach((video) => {
        video.style.width = `${100 / size}%`;
        video.style.height = `${100 / size}%`;
        video.style.left = `${(-x / size) * 100}%`;
        video.style.top = `${(-y / size) * 100}%`;
      });
    }

    function updateLens(clientX, clientY) {
      const rect = stage.getBoundingClientRect();
      const centerX = (clientX - rect.left) / rect.width;
      const centerY = (clientY - rect.top) / rect.height;
      state.lens.x = clamp(centerX - state.lens.size / 2, 0, 1 - state.lens.size);
      state.lens.y = clamp(centerY - state.lens.size / 2, 0, 1 - state.lens.size);
      renderLens();
    }

    root.querySelectorAll("[data-scene]").forEach((button) => {
      button.addEventListener("click", () => selectScene(button.dataset.scene));
    });
    root.querySelectorAll("[data-scene-dot]").forEach((button) => {
      button.addEventListener("click", () => selectScene(button.dataset.sceneDot));
    });
    root.querySelector("[data-scene-prev]").addEventListener("click", () => stepScene(-1));
    root.querySelector("[data-scene-next]").addEventListener("click", () => stepScene(1));

    root.querySelectorAll("[data-method]").forEach((button) => {
      button.addEventListener("click", () => {
        state.method = button.dataset.method;
        root.querySelectorAll("[data-method]").forEach((item) => item.classList.toggle("active", item === button));
        loadSelection();
      });
    });

    divider.addEventListener("pointerdown", (event) => {
      state.dragging = "split";
      stage.setPointerCapture(event.pointerId);
      updateSplit(event.clientX);
    });
    divider.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      const direction = event.key === "ArrowLeft" ? -2 : 2;
      const rect = stage.getBoundingClientRect();
      updateSplit(rect.left + ((state.split + direction) / 100) * rect.width);
    });

    lens.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      state.dragging = "lens";
      stage.setPointerCapture(event.pointerId);
      updateLens(event.clientX, event.clientY);
    });
    stage.addEventListener("pointermove", (event) => {
      if (state.dragging === "split") updateSplit(event.clientX);
      if (state.dragging === "lens") updateLens(event.clientX, event.clientY);
    });
    stage.addEventListener("pointerup", (event) => {
      state.dragging = null;
      if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
    });
    stage.addEventListener("pointercancel", () => {
      state.dragging = null;
    });

    detailToggle.addEventListener("change", () => {
      lens.hidden = !detailToggle.checked;
      detailGrid.hidden = !detailToggle.checked;
      lensSize.closest("label").hidden = !detailToggle.checked;
    });
    lensSize.addEventListener("input", () => {
      const size = Number(lensSize.value) / 100;
      state.lens.size = size;
      state.lens.x = clamp(state.lens.x, 0, 1 - size);
      state.lens.y = clamp(state.lens.y, 0, 1 - size);
      renderLens();
    });

    overlay.style.clipPath = "inset(0 50% 0 0)";
    divider.style.left = "50%";
    renderLens();
    loadSelection();
  }

  initEnhancementComparison();
  initReconstructionComparison();

  const copyBibtexButton = document.querySelector("[data-copy-bibtex]");
  const bibtexCode = document.querySelector("[data-bibtex-code]");
  if (copyBibtexButton && bibtexCode) {
    copyBibtexButton.addEventListener("click", async () => {
      const label = copyBibtexButton.querySelector("span");
      try {
        const text = bibtexCode.textContent.trim();
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(text);
        } else {
          const textarea = document.createElement("textarea");
          textarea.value = text;
          textarea.style.position = "fixed";
          textarea.style.opacity = "0";
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand("copy");
          textarea.remove();
        }
        label.textContent = "Copied";
        window.setTimeout(() => { label.textContent = "Copy"; }, 1600);
      } catch (_error) {
        label.textContent = "Copy failed";
        window.setTimeout(() => { label.textContent = "Copy"; }, 1600);
      }
    });
  }

  const scrollTopButton = document.querySelector("[data-scroll-top]");
  if (scrollTopButton) {
    const updateScrollTopButton = () => {
      scrollTopButton.classList.toggle("visible", window.scrollY > 420);
    };
    scrollTopButton.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
    window.addEventListener("scroll", updateScrollTopButton, { passive: true });
    updateScrollTopButton();
  }
})();
