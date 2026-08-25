#!/usr/bin/env python3
# coding: utf-8

"""Prepare a BIDS cohort for LST-AI training.

Discovers sessions by globbing the FLAIR suffix, derives the matching T1w and
lesion mask from the same session prefix, and runs preprocess_session on each.
Writes one <session_id>/ directory per session under --output, plus a manifest CSV.

Expected layout (every suffix is configurable):

    <bids_root>/sub-X/ses-Y/anat/sub-X_ses-Y_FLAIR.nii.gz
    <bids_root>/sub-X/ses-Y/anat/sub-X_ses-Y_T1w.nii.gz
    <mask_root>/sub-X/ses-Y/anat/sub-X_ses-Y_space-FLAIR_label-lesion_mask.nii.gz

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
    """
    Return a dict with the manifest columns pre-filled for this session.

    Parameters
    ----------
    session_id : str
        The session identifier, e.g. sub-01_ses-01.
    flair : str
        Path to the FLAIR image.
    t1 : str or None
        Path to the T1w image, or None if not used.
    mask : str
        Path to the lesion mask.
    complete : str
        'yes' if all required files exist, 'no' otherwise.
    prepared : str, optional
        'yes' if the session has already been processed, 'no' otherwise (default: 'no').
    
    Returns
    -------
    dict
        A dictionary with keys corresponding to MANIFEST_FIELDS, pre-filled with the
        provided values and empty strings for the output fields.

    """
    row = {k: '' for k in MANIFEST_FIELDS}
    row.update(session_id=session_id, complete=complete, prepared=prepared,
               flair_in=flair, t1_in=t1 or '', mask_in=mask)
    return row


def lesion_mm3(path):
    """
    Calculates the lesion volume in mm3. 
    The function loads a NIfTI file from the given path, counts the number of voxels that are greater than zero (indicating the presence of a lesion), 
    and multiplies this count by the volume of a single voxel (derived from the image header) to obtain the total lesion volume in cubic millimeters.

    Parameters
    ----------
    path : str
        Path to the lesion mask NIfTI file.

    Returns
    -------
    float
        The volume of the lesion in cubic millimeters. If the mask is empty, returns 0.0.

    """
    img = nib.load(path)
    voxels = int((np.asarray(img.dataobj) > 0).sum())
    return voxels * float(np.prod(img.header.get_zooms()[:3]))


def lesion_check(row, mask_in, seg_out):
    """
    Conducts a lesion volume check and updates the manifest row with the results.
    The volume of the lesion in the input mask and the output segmentation is calculated,
    and the fraction of the lesion retained after processing is computed. The manifest row
    is updated with these values.

    Parameters
    ----------
    row : dict
        The manifest row to be updated with lesion volume information.
    mask_in : str
        Path to the input lesion mask NIfTI file.
    seg_out : str
        Path to the output segmentation NIfTI file.

    Returns
    -------
    tuple
        A tuple containing:
        - mm3_in (float): The volume of the lesion in the input mask in cubic millimeters.
        - mm3_out (float): The volume of the lesion in the output segmentation in cubic millimeters.
        - retained (float): The fraction of the lesion volume retained after processing. If the input mask is empty, this will be 1.0. 
    """
    mm3_in, mm3_out = lesion_mm3(mask_in), lesion_mm3(seg_out)
    retained = mm3_out / mm3_in if mm3_in else 1.0   # an empty mask loses nothing
    row.update(lesion_mm3_in=f"{mm3_in:.0f}",
               lesion_mm3_out=f"{mm3_out:.0f}",
               lesion_retained=f"{retained:.3f}")
    return mm3_in, mm3_out, retained


def find_sessions(bids_root, mask_root, flair_suffix, t1_suffix, mask_suffix, channels):
    """
    Finds all sessions in the BIDS dataset by globbing for FLAIR files with the specified suffix.
    For each FLAIR file found, it derives the corresponding T1w and lesion mask paths based on the session prefix and checks for their existence. 
    It returns a list of tuples containing the session ID, FLAIR path, T1w path (or None if not used), lesion mask path, and a list of any missing files.

    Parameters
    ----------
    bids_root : str
        The root directory of the BIDS dataset.
    mask_root : str
        The root directory to search for lesion masks. Defaults to bids_root if not specified.
    flair_suffix : str
        The suffix used to identify FLAIR files (e.g., 'FLAIR.nii.gz').
    t1_suffix : str
        The suffix used to identify T1w files (e.g., 'T1w.nii.gz').
    mask_suffix : str
        The suffix used to identify lesion mask files (e.g., 'label-lesion_mask.nii.gz').
    channels : int
        The number of channels to use (1 for FLAIR only, 2 for FLAIR + T1w). If channels is 1, the T1w path will be set to None.
    
    Returns
    -------
    list of tuples
        A list of tuples, each containing:
        - session_id (str): The session identifier (e.g., 'sub-01_ses-01').
        - flair (str): The path to the FLAIR image.
        - t1 (str or None): The path to the T1w image, or None if not used.
        - mask (str): The path to the lesion mask.
        - missing (list of str): A list of any missing files for the session. If all required files are present, this list will be empty.
    """
    flair_paths = sorted(glob.glob(os.path.join(bids_root, 'sub-*', 'ses-*', 'anat', f'*_{flair_suffix}')))
    if not flair_paths:   # sessionless BIDS
        flair_paths = sorted(glob.glob(os.path.join(bids_root, 'sub-*', 'anat', f'*_{flair_suffix}')))

    # Empty when nothing matched; the caller reports that.
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
    """
    Returns the expected output paths for a given session, based on the output directory, session ID, and number of channels.

    Parameters
    ----------
    output : str
        The parent directory for the per-session output directories.
    session_id : str
        The session identifier (e.g., 'sub-01_ses-01').
    channels : int
        The number of channels to use (1 for FLAIR only, 2 for FLAIR + T1w). If channels is 1, the T1w output path will be set to None.
    
    Returns
    -------
    dict
        A dictionary containing the expected output paths for the FLAIR, T1w (if applicable), and segmentation files, with keys 'flair', 't1', and 'seg'.
    """
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
    parser.add_argument('--mask_suffix', default='space-FLAIR_label-lesion_mask.nii.gz', type=str,
                        help='Everything after "<session_id>_" in the lesion mask filename.')

    parser.add_argument('--manifest', default=None, type=str,
                        help='Manifest CSV path (default: <output>/prepare_cohort_manifest.csv).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Reprocess sessions whose outputs already exist.')
    parser.add_argument('--dry_run', action='store_true',
                        help='List the resolved inputs and output names, then exit. '
                             'Use it to check the suffixes: a wrong one changes the '
                             'session id silently rather than failing.')
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

    if args.dry_run:
        for session_id, flair, t1, mask, _ in todo:
            print(f"  {session_id}")
            print(f"    flair {flair}")
            if t1:
                print(f"    t1    {t1}")
            print(f"    mask  {mask}")
            print(f"    out   {os.path.join(output, session_id)}{os.sep}")
        print(f"\nDry run: {len(todo)} session(s) would be processed. Nothing written.")
        return 0

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
            except Exception as exc:
                # One greppable line; the full traceback goes to stderr below.
                frame = traceback.extract_tb(exc.__traceback__)[-1]
                row['error'] = (f"{type(exc).__name__}: {exc} "
                                f"[{os.path.basename(frame.filename)}:{frame.lineno}]"
                                ).replace('\n', ' ')[:300]
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
