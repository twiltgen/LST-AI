#!/usr/bin/env python3
# coding: utf-8

"""Prepare a BIDS cohort for LST-AI training.

Discovers sessions by globbing the FLAIR suffix, derives the matching T1w and
lesion mask from the same session prefix, and runs preprocess_session on each.
Writes one <session_id>/ directory per session under --output, plus a manifest CSV.

Expected layout (every suffix is configurable):

    <bids_root>/sub-X/ses-Y/anat/sub-X_ses-Y_space-mni_FLAIR.nii.gz
    <bids_root>/sub-X/ses-Y/anat/sub-X_ses-Y_space-mni_T1w.nii.gz
    <mask_root>/sub-X/ses-Y/anat/sub-X_ses-Y_space-mni_label-lesion_mask.nii.gz

--mask_root defaults to --bids_root; point it at a derivatives pipeline when the
masks live there, as in derivatives/manual_segmentation.

Sessions run one at a time: HD-BET wants the whole GPU, and greedy is already
threaded via --threads.
"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback

import nibabel as nib
import numpy as np

from preprocess_session import preprocess_session

MANIFEST_FIELDS = ['session_id', 'complete', 'prepared', 'flair_in', 't1_in', 'mask_in',
                   'flair_out', 't1_out', 'seg_out', 'lesion_mm3_in', 'lesion_mm3_out',
                   'lesion_retained', 'error']


def manifest_row(session_id, flair, t1, mask, complete, prepared='no'):
    """A row with the input side filled in; the caller adds the output side."""
    row = {k: '' for k in MANIFEST_FIELDS}
    row.update(session_id=session_id, complete=complete, prepared=prepared,
               flair_in=flair, t1_in=t1 or '', mask_in=mask)
    return row


def lesion_mm3(path):
    """Lesion volume in mm3. dataobj avoids the float64 copy get_fdata() makes."""
    img = nib.load(path)
    voxels = int((np.asarray(img.dataobj) > 0).sum())
    return voxels * float(np.prod(img.header.get_zooms()[:3]))


def lesion_check(row, mask_in, seg_out):
    """Fill the sanity-check columns and return (mm3 in, mm3 out, retained).

    A mask that was not in native FLAIR space warps to a valid but near-empty
    file, so compare the volume rather than trusting that the warp succeeded.
    """
    mm3_in, mm3_out = lesion_mm3(mask_in), lesion_mm3(seg_out)
    retained = mm3_out / mm3_in if mm3_in else 0.0
    row.update(lesion_mm3_in=f"{mm3_in:.0f}",
               lesion_mm3_out=f"{mm3_out:.0f}",
               lesion_retained=f"{retained:.3f}")
    return mm3_in, mm3_out, retained


def find_sessions(bids_root, mask_root, flair_suffix, t1_suffix, mask_suffix, channels):
    """Return one (id, flair, t1 or None, mask, missing) per FLAIR found.

    ``missing`` lists the siblings that do not exist, so an incomplete session is
    still reported rather than dropped. Globs sub-*/ses-*/anat explicitly rather
    than recursively, so a derivatives/ tree inside the BIDS root is never
    mistaken for subject data.
    """
    flair_paths = sorted(glob.glob(os.path.join(bids_root, 'sub-*', 'ses-*', 'anat', f'*_{flair_suffix}')))
    if not flair_paths:   # sessionless BIDS
        flair_paths = sorted(glob.glob(os.path.join(bids_root, 'sub-*', 'anat', f'*_{flair_suffix}')))

    sessions = []
    for flair in flair_paths:
        # The session id is the filename with the suffix stripped; the siblings are
        # then that id plus their own suffix, in the same anat/ dir under each root.
        anat_dir = os.path.dirname(flair)
        session_id = os.path.basename(flair)[: -len(f"_{flair_suffix}")]
        anat_rel = os.path.relpath(anat_dir, bids_root)
        mask = os.path.join(mask_root, anat_rel, f"{session_id}_{mask_suffix}")
        t1 = os.path.join(anat_dir, f"{session_id}_{t1_suffix}") if channels == 2 else None

        missing = [p for p in ([mask, t1] if t1 else [mask]) if not os.path.exists(p)]
        sessions.append((session_id, flair, t1, mask, missing))
    return sessions


def expected_outputs(output, session_id, channels):
    """The files preprocess_session writes, keyed as it keys them."""
    session_dir = os.path.join(output, session_id)
    paths = {n: os.path.join(session_dir, f"{session_id}_{n}.nii.gz") for n in ('flair', 'seg')}
    paths['t1'] = os.path.join(session_dir, f"{session_id}_t1.nii.gz") if channels == 2 else None
    return paths


def main():
    parser = argparse.ArgumentParser(
        description='Prepare a BIDS cohort for LST-AI training.')

    parser.add_argument('--bids_root', required=True, type=str,
                        help='Root of the BIDS dataset (contains sub-*/).')
    parser.add_argument('--output', required=True, type=str,
                        help='Parent directory for the per-session output directories.')
    parser.add_argument('--channels', required=True, type=int, choices=(1, 2),
                        help='2: FLAIR + T1w. 1: FLAIR only (--t1_suffix unused).')

    parser.add_argument('--mask_root', default=None, type=str,
                        help='Root to search for lesion masks (default: --bids_root). '
                             'Set to e.g. <bids_root>/derivatives/manual_segmentation.')

    parser.add_argument('--flair_suffix', default='FLAIR.nii.gz', type=str,
                        help='Everything after "<session_id>_" in the FLAIR filename.')
    parser.add_argument('--t1_suffix', default='T1w.nii.gz', type=str,
                        help='Everything after "<session_id>_" in the T1w filename.')
    parser.add_argument('--mask_suffix', default='label-lesion_mask.nii.gz', type=str,
                        help='Everything after "<session_id>_" in the lesion mask filename.')

    parser.add_argument('--manifest', default=None, type=str,
                        help='Manifest CSV path (default: <output>/prepare_cohort_manifest.csv).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Reprocess sessions whose outputs already exist.')
    parser.add_argument('--min_lesion_retention', default=0.8, type=float,
                        help='Warn when the warped mask keeps less than this fraction of '
                             'its original volume (default: 0.8).')

    parser.add_argument('--fast-mode', action='store_true', dest='fast',
                        help='Only use one model for hd-bet.')
    parser.add_argument('--device', default='0', type=str,
                        help='Either int for GPU ID or "cpu" for CPU (default: 0)')
    parser.add_argument('--threads', default=1, type=int,
                        help='Threads for registration (default: 1).')

    args = parser.parse_args()

    # Resolve roots up front so the manifest records absolute paths.
    bids_root = os.path.abspath(args.bids_root)
    mask_root = os.path.abspath(args.mask_root) if args.mask_root else bids_root
    output = os.path.abspath(args.output)

    assert os.path.isdir(bids_root), f'BIDS root does not exist: {bids_root}'
    assert os.path.isdir(mask_root), f'Mask root does not exist: {mask_root}'
    assert not os.path.isfile(output), 'Please provide an output path, not a filename.'

    # every session found is reported, complete or not
    sessions = find_sessions(bids_root=bids_root, 
                             mask_root=mask_root, 
                             flair_suffix=args.flair_suffix,
                             t1_suffix=args.t1_suffix, 
                             mask_suffix=args.mask_suffix, 
                             channels=args.channels)
    complete = [s for s in sessions if not s[4]]
    incomplete = [s for s in sessions if s[4]]

    print(f"BIDS root : {bids_root}")
    print(f"Mask root : {mask_root}")
    print(f"Output    : {output}")
    print(f"Mode      : {args.channels}-channel")
    print(f"Found     : {len(complete)} complete session(s), {len(incomplete)} incomplete session(s)\n")

    for session_id, _, _, _, missing in incomplete:
        print(f"  skip {session_id}: missing: {', '.join(missing)}")
    if incomplete:
        print()

    if not sessions:
        print("Nothing found. Check --bids_root and --flair_suffix against the dataset.")
        return 1

    # split the complete sessions into those already prepared and those to process
    prepared, todo = [], []
    for session in complete:
        outputs = expected_outputs(output, session[0], args.channels)
        if all(os.path.exists(p) for p in outputs.values() if p) and not args.overwrite:
            print(f"  skip {session[0]}: already prepared (use --overwrite to redo)")
            prepared.append((session, outputs))
        else:
            todo.append(session)
    if prepared:
        print()

    # create the output directory and manifest path, then process each session
    os.makedirs(output, exist_ok=True)
    manifest_path = args.manifest or os.path.join(output, 'prepare_cohort_manifest.csv')

    # One session at a time; see the module docstring for why.
    n_ok = n_failed = n_warned = 0
    started = time.time()

    # Written from scratch each run, one row per session found. Row-by-row with a
    # flush, so a run that is killed still leaves a usable manifest.
    with open(manifest_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        # The sessions this run will not touch, recorded before any processing starts.
        for session_id, flair, t1, mask, missing in incomplete:
            writer.writerow(manifest_row(session_id, flair, t1, mask, complete='no'))

        for (session_id, flair, t1, mask, _), outputs in prepared:
            row = manifest_row(session_id, flair, t1, mask, complete='yes', prepared='yes')
            row.update(flair_out=outputs['flair'], 
                       seg_out=outputs['seg'],
                       t1_out=outputs['t1'] or '')
            # Re-checked from disk, so the column is filled for every prepared session.
            try:
                _, _, retained = lesion_check(row, mask, outputs['seg'])
                if retained < args.min_lesion_retention:
                    n_warned += 1
                    print(f"  WARNING: {session_id} kept only {retained:.1%} of its lesion volume.")
            except Exception as exc:
                row['error'] = f"lesion check failed: {exc}"
            writer.writerow(row)
        fh.flush()

        for i, (session_id, flair, t1, mask, _) in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {session_id}")
            # Pre-filled so a failed session still records what it was given.
            row = manifest_row(session_id, flair, t1, mask, complete='yes')
            t0 = time.time()
            try:
                written = preprocess_session(flair=flair,
                                             gt_seg=mask,
                                             output=output,
                                             session_id=session_id,
                                             t1=t1,
                                             fast=args.fast,
                                             device=args.device,
                                             threads=args.threads)

                row.update(prepared='yes',
                           flair_out=written['flair'], 
                           seg_out=written['seg'],
                           t1_out=written['t1'] or '')
                mm3_in, mm3_out, retained = lesion_check(row, mask, written['seg'])
                n_ok += 1
                print(f"  ok  {time.time() - t0:.0f}s  "
                      f"lesion {mm3_in:.0f} -> {mm3_out:.0f} mm3 ({retained:.1%} retained)")
                if retained < args.min_lesion_retention:
                    n_warned += 1
                    print(f"  WARNING: only {retained:.1%} of the lesion volume survived the "
                          f"warp. Is {os.path.basename(mask)} really in native FLAIR space?")

            # One bad session must not end the cohort run. KeyboardInterrupt still stops it.
            except Exception:
                row['error'] = traceback.format_exc(limit=1).strip().replace('\n', ' ')
                n_failed += 1
                print(f"  FAILED after {time.time() - t0:.0f}s", file=sys.stderr)
                traceback.print_exc()

            writer.writerow(row)
            fh.flush()

    print(f"\n{'=' * 60}")
    print(f"Prepared {n_ok}/{len(todo)} session(s) in {(time.time() - started) / 60:.1f} min")
    if prepared:
        print(f"{len(prepared)} session(s) were already prepared.")
    if incomplete:
        print(f"{len(incomplete)} session(s) are incomplete -- see the 'complete' column.")
    if n_warned:
        print(f"{n_warned} session(s) lost lesion volume in the warp -- check the manifest.")
    if n_failed:
        print(f"{n_failed} session(s) failed -- see the 'error' column in the manifest.")
    if not complete:
        print("No complete session. Check --mask_root and --mask_suffix against the dataset.")
    print(f"Manifest: {manifest_path}")
    print(f"Train on: {output}")
    return 1 if n_failed or not complete else 0


if __name__ == "__main__":
    sys.exit(main())
