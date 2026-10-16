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
import os
import sys
import time
import traceback

# common.py sits one level up, beside the training/ and inference/ pipelines.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import find_sessions, lesion_check
from preprocess_session import preprocess_session

MANIFEST_FIELDS = ['session_id', 'complete', 'prepared', 'flair_in', 't1_in', 'mask_in',
                   'flair_out', 't1_out', 'seg_out', 'lesion_mm3_in', 'lesion_mm3_out',
                   'lesion_retained', 'error']


def manifest_row(session, complete, prepared='no'):
    """
    Return a dict with the manifest columns pre-filled for this session.

    Parameters
    ----------
    session : common.Session
        The discovered session, carrying its id and its input paths.
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
    row.update(session_id=session.session_id, complete=complete, prepared=prepared,
               flair_in=session.flair, t1_in=session.t1 or '', mask_in=session.mask)
    return row


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

    parser.add_argument('--stripped', action='store_true', dest='stripped',
                        help='Images are already skull stripped. Skip skull-stripping.')
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

    # every session found is reported, complete or not; training needs the ground truth
    sessions = find_sessions(bids_root=bids_root,
                             mask_root=mask_root,
                             flair_suffix=args.flair_suffix,
                             t1_suffix=args.t1_suffix,
                             mask_suffix=args.mask_suffix,
                             channels=args.channels,
                             require_mask=True)
    complete = [s for s in sessions if not s.missing]
    incomplete = [s for s in sessions if s.missing]

    print(f"BIDS root : {bids_root}")
    print(f"Mask root : {mask_root}")
    print(f"Output    : {output}")
    print(f"Mode      : {args.channels}-channel")
    print(f"Stripping : {'skipped (inputs already skull-stripped)' if args.stripped else 'HD-BET'}")
    print(f"Found     : {len(complete)} complete session(s), {len(incomplete)} incomplete session(s)\n")

    for session in incomplete:
        print(f"  skip {session.session_id}: missing: {', '.join(session.missing)}")
    if incomplete:
        print()

    if not sessions:
        print("Nothing found. Check --bids_root and --flair_suffix against the dataset.")
        return 1

    # split the complete sessions into those already prepared and those to process
    prepared, todo = [], []
    for session in complete:
        outputs = expected_outputs(output, session.session_id, args.channels)
        if all(os.path.exists(p) for p in outputs.values() if p) and not args.overwrite:
            print(f"  skip {session.session_id}: already prepared (use --overwrite to redo)")
            prepared.append((session, outputs))
        else:
            todo.append(session)
    if prepared:
        print()

    if args.dry_run:
        for session in todo:
            print(f"  {session.session_id}")
            print(f"    flair {session.flair}")
            if session.t1:
                print(f"    t1    {session.t1}")
            print(f"    mask  {session.mask}")
            print(f"    out   {os.path.join(output, session.session_id)}{os.sep}")
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
        for session in incomplete:
            writer.writerow(manifest_row(session, complete='no'))

        for session, outputs in prepared:
            row = manifest_row(session, complete='yes', prepared='yes')
            row.update(flair_out=outputs['flair'],
                       seg_out=outputs['seg'],
                       t1_out=outputs['t1'] or '')
            # Re-checked from disk, so the column is filled for every prepared session.
            try:
                _, _, retained = lesion_check(row, session.mask, outputs['seg'])
                if retained < args.min_lesion_retention:
                    n_warned += 1
                    print(f"  WARNING: {session.session_id} kept only {retained:.1%} of its lesion volume.")
            except Exception as exc:
                row['error'] = f"lesion check failed: {exc}"
            writer.writerow(row)
        fh.flush()

        for i, session in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {session.session_id}")
            # Pre-filled so a failed session still records what it was given.
            row = manifest_row(session, complete='yes')
            t0 = time.time()
            try:
                written = preprocess_session(flair=session.flair,
                                             gt_seg=session.mask,
                                             output=output,
                                             session_id=session.session_id,
                                             stripped=args.stripped,
                                             t1=session.t1,
                                             fast=args.fast,
                                             device=args.device,
                                             threads=args.threads)

                row.update(prepared='yes',
                           flair_out=written['flair'],
                           seg_out=written['seg'],
                           t1_out=written['t1'] or '')
                mm3_in, mm3_out, retained = lesion_check(row, session.mask, written['seg'])
                n_ok += 1
                print(f"  ok  {time.time() - t0:.0f}s  "
                      f"lesion {mm3_in:.0f} -> {mm3_out:.0f} mm3 ({retained:.1%} retained)")
                if retained < args.min_lesion_retention:
                    n_warned += 1
                    print(f"  WARNING: only {retained:.1%} of the lesion volume survived the "
                          f"warp. Is {os.path.basename(session.mask)} really in native FLAIR space?")

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
