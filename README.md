# 3D-USE Project Page

Static project page for **3D-USE: From Image-Level to Scene-Level Underwater Enhancement**.

## Preview

Open `index.html` directly or serve this directory with any static web server. No build step or package installation is required.

## Assets

Place the final teaser and pipeline PNGs under `media/figures/`. Reconstruction and enhancement videos use parallel layouts:

```text
media/{reconstruction,enhancement}/
├── iui3/
│   ├── watersplatting.mp4
│   ├── seasplat.mp4
│   ├── marinestd-gs.mp4
│   ├── plenodium.mp4
│   ├── 3d-uir.mp4
│   └── 3d-use.mp4
├── curasao/
├── japanese/
├── panama/
└── d3/
    └── ...same filenames...
```

Keep every video for a scene at the same resolution, frame rate, frame count, and camera path so paired playback stays synchronized.
