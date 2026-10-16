#!/usr/bin/env python3
# coding: utf-8

"""Write out the deterministic loader preprocessing (stage B) for visual inspection.

Applies what lst_training.data.MSDataset does to every volume before augmentation,
using the loader's own functions, and saves each step as a NIfTI:

    B1  centre-crop to --shape        <id>_desc-cropped_FLAIR / _T1w
                                      <id>_label-lesion_desc-cropped_mask
    B2  brain mask := flair != 0      <id>_label-brain_desc-cropped_mask
    B3  binarise the lesion mask      <id>_label-lesion_desc-croppedBinarized_mask
    B4  clip to in-brain [0.5, 99.5]  <id>_desc-croppedNormalized_FLAIR / _T1w
        percentiles, min-max to
        [0, 1], zero outside brain

Inputs must already be skull-stripped and in MNI space (the output of
prepare_data/training/prepare_cohort.py, or equivalent). Outputs go to
<bids_root>/derivatives/training_preprocessing/sub-X/ses-Y/anat/, with the
affine shifted by the crop so they overlay on the inputs.

    python tools/training_preprocessing.py --bids_root /data/cohort --channels 2
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)
sys.path.insert(0, os.path.join(TRAINING, 'prepare_data'))

from common import find_sessions
from lst_training.data import adapt_shape, crop_slices, normalise_modality


def cropped_affine(affine, slices):
    """Shift the origin to the first voxel of the crop window."""
    out = affine.copy()
    out[:3, 3] = affine[:3, :3] @ np.array([s.start for s in slices]) + affine[:3, 3]
    return out


def save(data, affine, path, dtype):
    img = nib.Nifti1Image(data.astype(dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    nib.save(img, path)


def process_session(session, out_dir, shape, channels):
    sid = session.session_id
    flair_img = nib.load(session.flair)
    slices = crop_slices(flair_img.shape, shape)
    affine = cropped_affine(flair_img.affine, slices)

    inputs = {'FLAIR': flair_img, 'mask': nib.load(session.mask)}
    if channels == 2:
        inputs['T1w'] = nib.load(session.t1)
    for name, img in inputs.items():
        # the loader crops every volume with the FLAIR's window, so they must share a grid
        if img.shape[:3] != flair_img.shape[:3]:
            raise ValueError(f"{name} shape {img.shape} != FLAIR shape {flair_img.shape}")

    def out(name):
        return os.path.join(out_dir, f"{sid}_{name}.nii.gz")

    # B1
    flair = adapt_shape(flair_img.get_fdata(), shape)
    gt = adapt_shape(inputs['mask'].get_fdata(), shape)
    save(flair, affine, out('desc-cropped_FLAIR'), np.float32)
    save(gt, affine, out('label-lesion_desc-cropped_mask'), np.float32)

    # B2, B3
    brain = np.zeros(flair.shape, np.float32)
    brain[flair != 0] = 1.0
    gt = (gt > 0).astype(np.float32)
    save(brain, affine, out('label-brain_desc-cropped_mask'), np.uint8)
    save(gt, affine, out('label-lesion_desc-croppedBinarized_mask'), np.uint8)

    # B4
    mods = {'FLAIR': flair}
    if channels == 2:
        mods['T1w'] = adapt_shape(inputs['T1w'].get_fdata(), shape)
        save(mods['T1w'], affine, out('desc-cropped_T1w'), np.float32)
    report = []
    for name, vol in mods.items():
        lo, hi = np.percentile(vol[brain != 0], (0.5, 99.5))
        save(normalise_modality(vol, brain), affine, out(f'desc-croppedNormalized_{name}'), np.float32)
        report.append(f"{name} clip [{lo:.1f}, {hi:.1f}]")

    brain_frac = brain.mean()
    print(f"  ok  brain {brain_frac:.1%} of crop, lesion {int(gt.sum())} vox, " + ", ".join(report))
    # an unstripped head fills most of the crop, so flair != 0 would not be a brain mask
    if brain_frac > 0.5:
        print("  WARNING: over half the crop is non-zero -- is this FLAIR skull-stripped?")


def main():
    parser = argparse.ArgumentParser(
        description='Write the LST-AI loader preprocessing (stage B) of a BIDS cohort as NIfTIs.')
    parser.add_argument('--bids_root', required=True, type=str,
                        help='Root of the BIDS dataset (contains sub-*/).')
    parser.add_argument('--channels', required=True, type=int, choices=(1, 2),
                        help='2: FLAIR + T1w. 1: FLAIR only (--t1_suffix unused).')
    parser.add_argument('--mask_root', default=None, type=str,
                        help='Root to search for lesion masks (default: --bids_root).')
    parser.add_argument('--output', default=None, type=str,
                        help='Default: <bids_root>/derivatives/training_preprocessing.')
    parser.add_argument('--flair_suffix', default='FLAIR.nii.gz', type=str)
    parser.add_argument('--t1_suffix', default='T1w.nii.gz', type=str)
    parser.add_argument('--mask_suffix', default='space-FLAIR_label-lesion_mask.nii.gz', type=str)
    parser.add_argument('--shape', type=int, nargs=3, default=(192, 192, 192),
                        help='Crop size, as --shape in lst_training.train.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Reprocess sessions whose outputs already exist.')
    args = parser.parse_args()

    bids_root = os.path.abspath(args.bids_root)
    mask_root = os.path.abspath(args.mask_root) if args.mask_root else bids_root
    output = os.path.abspath(args.output or os.path.join(bids_root, 'derivatives',
                                                         'training_preprocessing'))
    shape = tuple(args.shape)

    sessions = find_sessions(bids_root=bids_root, mask_root=mask_root,
                             flair_suffix=args.flair_suffix, t1_suffix=args.t1_suffix,
                             mask_suffix=args.mask_suffix, channels=args.channels,
                             require_mask=True)
    if not sessions:
        print("Nothing found. Check --bids_root and --flair_suffix against the dataset.")
        return 1
    print(f"Output: {output}\nFound {len(sessions)} session(s)\n")

    os.makedirs(output, exist_ok=True)
    description = os.path.join(output, 'dataset_description.json')
    if not os.path.exists(description):
        with open(description, 'w') as fh:
            json.dump({'Name': 'training_preprocessing', 'BIDSVersion': '1.9.0',
                       'DatasetType': 'derivative',
                       'GeneratedBy': [{'Name': 'LST-AI training_preprocessing.py',
                                        'Description': f'MSDataset stage B, crop {list(shape)}'}]},
                      fh, indent=2)

    n_ok = n_failed = n_skipped = 0
    for i, session in enumerate(sessions, 1):
        print(f"[{i}/{len(sessions)}] {session.session_id}")
        if session.missing:
            print(f"  skip: missing {', '.join(session.missing)}")
            n_skipped += 1
            continue
        out_dir = os.path.join(output, os.path.relpath(os.path.dirname(session.flair), bids_root))
        done = os.path.join(out_dir, f"{session.session_id}_desc-croppedNormalized_FLAIR.nii.gz")
        if os.path.exists(done) and not args.overwrite:
            print("  skip: already done (use --overwrite to redo)")
            n_skipped += 1
            continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            process_session(session, out_dir, shape, args.channels)
            n_ok += 1
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_failed += 1

    print(f"\n{n_ok} done, {n_skipped} skipped, {n_failed} failed. Results in {output}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
