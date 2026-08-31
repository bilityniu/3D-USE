# Data preparation

## Scene layout

3D-USE consumes a COLMAP reconstruction and one pseudo-depth image for every
training image:

```text
SCENE/
├── images_wb/
│   ├── frame_0001.png
│   └── ...
├── colmap/sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   └── points3D.bin
└── depth/
    ├── frame_0001.png
    ├── ...
    └── manifest.json
```

Image and pseudo-depth stems must match. Alternative directory names can be
passed to the training recipes with `--images-path`, `--depths-path`, and
`--colmap-path`.

## Depth Anything V2 pseudo-depth

Obtain an official Depth Anything V2 checkout and a checkpoint whose encoder
matches `--encoder`. Generate the prior with:

```bash
python3 scripts/generate_da2_pseudodepth.py \
  --images /path/to/SCENE/images_wb \
  --output /path/to/SCENE/depth \
  --checkpoint /path/to/depth_anything_v2_vitl.pth \
  --da2-repo /path/to/Depth-Anything-V2 \
  --encoder vitl
```

The script saves 16-bit single-channel PNGs plus a provenance manifest. These
values are a relative, disparity-like geometry prior and are not metric depth.
Stage 1 uses correlation-based alignment, so no cross-image metric scale is
assumed.

## Paired 2D transition data

Calibrator training requires two filename-aligned directories:

```text
PAIRS/
├── raw/
│   ├── image_0001.png
│   └── ...
└── enhanced/
    ├── image_0001.png
    └── ...
```

The enhanced images define paired appearance transitions; they are not copied
into a target 3D scene and are never used as its per-view RGB labels.

Compile the pairs into transition observations:

```bash
python scripts/build_transition_dataset.py \
  --raw-dir /path/to/PAIRS/raw \
  --target-dir /path/to/PAIRS/enhanced \
  --uie-proposer-checkpoint weights/uie_proposer.pth \
  --output /path/to/paired_transitions.pt \
  --dataset-name paired_uie \
  --target-provenance your_target_source
```

The frozen UIE proposer is used here only to generate proposal operators. The output caches both proposal-conditioned observations and paired target operators, so the proposer is not loaded again by the calibrator trainer. It also stores the transition-extraction configuration and source metadata. Use data that are disjoint from all target 3D scenes and evaluation views.
