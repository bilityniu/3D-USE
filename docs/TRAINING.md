# Training

## Complete two-stage run

```bash
bash recipes/run_two_stage.sh \
  --data /path/to/SCENE \
  --output /path/to/outputs/SCENE \
  --calibrator weights/transition_calibrator.pth \
  --uie-proposer weights/uie_proposer.pth
```

The command trains Stage 1 for 15K iterations and Stage 2 for another 5K
iterations. Existing checkpoints are reused only when `--reuse-existing` is
passed explicitly; incomplete runs are not overwritten.

Expected outputs include:

```text
outputs/SCENE/
├── stage1/3duse-stage1/run/
│   ├── config.yml
│   └── nerfstudio_models/step-000015000.ckpt
└── stage2/3duse-stage2/run/
    ├── config.yml
    └── nerfstudio_models/step-000020000.ckpt
```

## Separate stages

Stage 1:

```bash
bash recipes/train_stage1.sh \
  --data /path/to/SCENE \
  --output /path/to/outputs/SCENE
```

Stage 2:

```bash
bash recipes/train_stage2.sh \
  --data /path/to/SCENE \
  --output /path/to/outputs/SCENE \
  --stage1-checkpoint-dir \
    /path/to/outputs/SCENE/stage1/3duse-stage1/run/nerfstudio_models \
  --calibrator weights/transition_calibrator.pth \
  --uie-proposer weights/uie_proposer.pth
```

Run either script with `--help` for alternative data paths, step budgets,
logging backends, and executable overrides.

## Train the transition calibrator

The released checkpoints can be used directly. To retrain the calibrator, first build an operator dataset from filename-aligned UIEB underwater/reference pairs. In this offline step, the frozen UIE proposer provides the proposal transition, while the paired reference provides its calibration target:

```bash
python scripts/build_transition_dataset.py \
  --raw-dir /path/to/UIEB/raw \
  --target-dir /path/to/UIEB/reference \
  --uie-proposer-checkpoint weights/uie_proposer.pth \
  --output /path/to/uieb_transitions.pt \
  --dataset-name UIEB \
  --target-provenance paired-reference
```

The resulting file already contains the cached proposal and target operators; the UIE proposer is therefore not loaded by the calibrator trainer. Train the calibrator from this compiled dataset:

```bash
python scripts/train_transition_calibrator.py \
  --dataset /path/to/uieb_transitions.pt \
  --output /path/to/transition_calibrator.pth
```

The single compiled file contains all paired transition observations. The default reproduces the released final fit: 750 optimization steps over all paired samples.

During Stage 2, the frozen proposer produces captured-view transitions and the calibrator maps them into the paired-data transition domain before ATC forms the fixed scene-global and Gaussian-local targets. Both checkpoints are used only for this target-construction step and are not loaded when rendering a trained scene.
