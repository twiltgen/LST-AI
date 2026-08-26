# Training LST-AI models

Trains the same network `lst_ai` runs at inference — `lst_ai.model.NNUNet3D` is imported,
not copied, so what you train is by construction what ships. That matters: the released
ensemble is *heterogeneous* (mdlA 28 filters, mdlB 24 with a single deep-supervision head,
mdlC 32), which is what happens when training and inference drift apart in separate places.

Ported from the original TensorFlow `train_model.py` / `data_loader.py`, and verified
against them rather than by inspection:

| | evidence |
|---|---|
| preprocessing | reproduces `data_loader.py` exactly (max abs diff **0.0**) |
| augmentation | matches under a shared RNG seed, **0.0** on every channel and target |
| losses | match the TF implementations to **1.9e-06** across six regimes |

## Install

```bash
pip install -e .                 # lst_ai itself, from the repository root
pip install -e training          # or just: pip install torch nibabel scipy scikit-image
```

## Data layout

One directory per subject, discovered by globbing for FLAIR:

```
<root>/**/<subject>_flair.nii.gz
<root>/**/<subject>_t1.nii.gz      # only for --in-channels 2
<root>/**/<subject>_seg.nii.gz     # binary lesion mask
```

Volumes must be skull-stripped and in MNI space — background is taken to be exactly zero,
and the brain mask is derived as `flair != 0`. Use `lst_ai`'s own registration and
stripping to prepare a cohort (the next section does exactly that for a BIDS dataset).

## Prepare a BIDS cohort

`prepare_training_data/prepare_cohort.py` turns a raw BIDS dataset into the layout above.
It globs `sub-*/ses-*/anat/` for the FLAIR, derives the T1w and lesion mask from the same
session prefix, registers to MNI, skull-strips with HD-BET, and warps the mask to match:

```bash
python prepare_cohort.py \
  --bids_root /data/cohort \
  --mask_root /data/cohort/derivatives/manual_segmentation \
  --output    data/train \
  --channels  2
```

Each filename is read as `<session_id>_<suffix>`, so `--flair_suffix`, `--t1_suffix` and
`--mask_suffix` describe the cohort and the session id is whatever the FLAIR suffix leaves
behind. `--mask_root` defaults to `--bids_root`; point it at a derivatives pipeline when the
masks live there. Run `--dry_run` first; a suffix that matches the wrong part of a filename
changes the session id rather than failing.

The ground truth must be in **native FLAIR space**; it is warped with the FLAIR affine. The
two modes are not interchangeable: with `--channels 2` the FLAIR reaches MNI through the T1
and is stripped with the T1's brain mask (identical to the 2-channel LST-AI workflow), with 
`--channels 1` the FLAIR goes straight to the atlas and is stripped directly. Prepare a cohort 
in the mode you intend to train.

Sessions run one at a time, a failure never stops the run, and re-running skips what is
already done (`--overwrite` redoes it). A manifest CSV in `--output` records every session
found along with its lesion volume before and after the warp: a mask that was not in FLAIR
space warps to a valid but near-empty file, so check the `lesion_retained` column rather
than trusting that the run succeeded. Then point `--train-data` straight at `--output`.
`preprocess_session.py` does the same work for a single session.

## Train

```bash
# dual-channel FLAIR + T1, the configuration the released models use
python -m lst_training.train --train-data data/train --val-data data/val --in-channels 2

# single-channel FLAIR only
python -m lst_training.train --train-data data/train --in-channels 1
```

Defaults reproduce the original recipe: SGD (momentum 0.9, Nesterov), lr 1e-2 on a cosine
schedule over `--epochs`, deep supervision at 1/2 and 1/4 resolution weighted 4/7 : 2/7 :
1/7, and checkpointing on the best *training* `out_seg` loss (as the original did, not on
validation). `--preset` selects one of the four loss combinations the original swept:
`nnUNet_bce-dice`, `nnUNet_bce-tversky`, `nnUNet_dsTversky`, `nnUNet_dsDice`.

Useful flags: `--filters`, `--conv-blocks`, `--bottleneck-filters` and `--ds-layers` to
reproduce a specific released variant; `--amp` for mixed precision on CUDA; `--no-augment`
to disable augmentation; `--shape` for a different crop; `--resume` to continue an
interrupted run.

Each invocation trains **one** model. The released ensemble is three, so reproducing it
means three runs (differing in topology and `--seed`) whose checkpoints are then averaged
at inference:

```bash
python -m lst_training.train --train-data data/train --name mdlA \
    --filters 28 --bottleneck-filters 168 --ds-layers -2 -3 --seed 0
python -m lst_training.train --train-data data/train --name mdlB \
    --filters 24 --bottleneck-filters 144 --ds-layers -2    --seed 1
python -m lst_training.train --train-data data/train --name mdlC \
    --filters 32 --ds-layers -2 -3 --seed 2
```

Run them sequentially. They must share the same `--in-channels`; inference refuses
a mixed ensemble.

### Resuming an interrupted run

Every epoch also writes `<out-dir>/UNet3D_MS_last_<name>.pt`, holding the model, 
optimiser, schedule, GradScaler, best loss so far and the history to date. 
Continue from it with **the same flags as the original run** and the `--resume` flag:

```bash
python -m lst_training.train <the original flags> \
    --resume checkpoints/UNet3D_MS_last_<name>.pt
```

Training picks up at the next epoch and the JSON history continues as one record rather than restarting.

### Monitoring a run

Every epoch appends to `<out-dir>/UNet3D_MS_final_<name>.json` — loss, per-head loss,
Dice, learning rate and wall time. That file always exists and needs no extra package.

For watching a long run, add `--tensorboard`:

```bash
pip install tensorboard
python -m lst_training.train --train-data data/train --tensorboard
tensorboard --logdir checkpoints/tb
```

Scalars are grouped as `train/…` and `val/…` so both splits of a metric share axes, with
each deep-supervision head logged separately — useful for spotting a head that has stopped
contributing. Pass `--tensorboard DIR` to choose the directory; the default is
`<out-dir>/tb/<name>`. Note this pulls in the standalone `tensorboard` package, which does
not depend on TensorFlow.

Input shape must be divisible by `2 ** conv_blocks`, and must leave more than one voxel in
the bottleneck — instance norm over a single voxel divides by `sqrt(eps)` and collapses the
block to its bias. `check_input_shape` rejects both cases with an explanatory error.

## Numerics: legacy vs modern

The released weights were trained under TensorFlow defaults that are **not** PyTorch's, and
`lst_ai.model` pins them so the shipped weights load correctly: LeakyReLU slope **0.3**
(torch: 0.01), instance-norm epsilon **1e-3** (torch: 1e-5), affine instance norm, no conv
bias, he_uniform init, and `K.epsilon()` = 1e-7 Dice smoothing.

Three of those are not quirks — a bias before an affine norm is redundant either way,
he_uniform is more principled than torch's legacy `kaiming_uniform(a=sqrt(5))`, and at
activation variance ~0.7 the epsilon shifts the denominator by 0.14%. **The slope is the
one real inherited accident**: 0.3 is a Keras default rather than a choice made for this
task, where nnU-Net and most of the literature use 0.01. If you are training fresh models
and do not need to ensemble them with the released ones, it is worth running both.

## Inference on a model you trained

```bash
python -m lst_training.inference --flair f.nii.gz --t1 t1.nii.gz \
    --checkpoints checkpoints/UNet3D_MS_final_<name>.pt --output seg.nii.gz \
    --intensity-range minus1-1 --mask-from flair
```

Those two flags matter. LST-AI's training and inference code disagree on preprocessing:
`data_loader.py` rescales to `[-1, 1]` and masks both modalities with the FLAIR mask, while
`segment.py` stops at `[0, 1]` and derives a mask per modality. A model trained here
follows the loader, so serving it needs the loader's conventions.

`--checkpoints` takes as many files as you like and averages their probabilities, so an
ensemble is just the three names listed above:

```bash
--checkpoints checkpoints/UNet3D_MS_final_mdl{A,B,C}.pt
```

Each `.pt` carries its own `config`, so members with different topologies load correctly
and need not match, only `--in-channels` has to agree across them.

## Tests

```bash
python -m pytest training/tests -q
```

The TensorFlow-derived fixtures can be regenerated with `training/tools/make_loss_reference.py`
and `training/tools/make_loader_reference.py` in a throwaway TF environment (TF 2.19 plus
`tf-keras`, since the released `.h5` need Keras 2).
