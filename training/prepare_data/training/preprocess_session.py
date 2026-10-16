#!/usr/bin/env python3
# coding: utf-8

"""Prepare a single session for LST-AI training.

Registers to MNI, skull-strips, and warps the ground truth lesion mask to match,
writing the volumes under the names lst_training.data.MSDataset globs for:

    <session_id>_flair.nii.gz
    <session_id>_t1.nii.gz      (2-channel only)
    <session_id>_seg.nii.gz

Two modes, matching the two deployment scenarios:

    2-channel   T1w -> MNI atlas, FLAIR -> MNI T1w, HD-BET on the T1w.
    1-channel   FLAIR -> MNI atlas directly, HD-BET on the FLAIR.

They do not yield interchangeable FLAIR volumes, so prepare a cohort in the mode
you intend to train.

The ground truth must be in native FLAIR space -- it is warped with the FLAIR
affine.

The work itself is in prepare_data/common.py, shared with the inference
preparation so that the two cannot drift apart.

REQUIRES
> greedy (picsl_greedy)
> HD-BET
> MNI atlas files (downloaded on first run)
"""
import os
import sys
import multiprocessing
import argparse

# common.py sits one level up, beside the training/ and inference/ pipelines.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import CITATION, preprocess_session as _preprocess_session


def preprocess_session(flair, gt_seg, output, session_id, stripped=False, t1=None,
                       fast=False, device='0', threads=None):
    """Preprocess one session for training. See common.preprocess_session; training
    always has a ground truth, and keeps no affine.

    Parameters
    ----------
    flair : str
        Path to the FLAIR image (zipped nifti).
    gt_seg : str
        Path to the ground truth segmentation in native FLAIR space (zipped nifti).
    output : str
        Path to the output directory where the processed images will be saved.
    session_id : str
        Session identifier used as the stem of the output filenames.
    stripped : bool
        If True, assumes the images are already skull-stripped and skips the skull-stripping step.
    t1 : str, optional
        Path to the T1w image (zipped nifti). Required for 2-channel mode. Default is None (1-channel mode).
    fast : bool, optional
        If True, uses a faster mode for HD-BET. Default is False.
    device : str, optional
        Device to use for processing. Can be an integer for GPU ID or "cpu" for CPU. Default is '0'.
    threads : int, optional
        Number of threads to be used for registration. If not provided, uses all available threads.

    Returns
    -------
    dict
        A dictionary containing the paths to the processed images:
        - 't1': Path to the processed T1w image (None in 1-channel mode).
        - 'flair': Path to the processed FLAIR image.
        - 'seg': Path to the processed ground truth segmentation.
    """
    assert gt_seg is not None, 'LST.AI aborted. Training requires a ground truth segmentation.'
    return _preprocess_session(flair=flair,
                               output=output,
                               session_id=session_id,
                               gt_seg=gt_seg,
                               stripped=stripped,
                               t1=t1,
                               fast=fast,
                               device=device,
                               threads=threads,
                               save_affine=False)


if __name__ == "__main__":

    print(CITATION)

    parser = argparse.ArgumentParser(description='Prepare data to train a new LST-AI model.')

    # Mode
    parser.add_argument('--channels',
                        dest='channels',
                        help='2: FLAIR + T1w (requires --t1). 1: FLAIR only.',
                        type=int,
                        choices=(1, 2),
                        required=True)

    # Input Images
    parser.add_argument('--t1',
                        dest='t1',
                        help='Path to T1 image (2-channel mode only)',
                        type=str,
                        default=None)
    parser.add_argument('--flair',
                        dest='flair',
                        help='Path to FLAIR image',
                        type=str,
                        required=True)
    parser.add_argument('--gt_seg',
                        dest='ground_truth_seg',
                        help='Path to ground truth segmentation',
                        type=str,
                        required=True)

    # Output Images
    parser.add_argument('--output',
                        dest='output',
                        help='Path to parent directory of session directories.',
                        type=str,
                        required=True)

    parser.add_argument('--session_id',
                        dest='session_id',
                        help='Session identifier used as the stem of the output filenames.',
                        type=str,
                        required=True)

    parser.add_argument('--stripped',
                        action='store_true',
                        dest='stripped',
                        help='Images are already skull stripped. Skip skull-stripping.')

    # Fast mode
    parser.add_argument('--fast-mode',
                        action='store_true',
                        dest='fast',
                        help='Only use one model for hd-bet.')

    # Computing Resources
    parser.add_argument('--device',
                        dest='device',
                        help='Either int for GPU ID or "cpu" for CPU (default: 0)',
                        type=str,
                        default='0')

    parser.add_argument('--threads',
                        dest='threads',
                        help='Number of threads to be used for registration (default: all available)',
                        type=int,
                        default=multiprocessing.cpu_count())

    args = parser.parse_args()

    # check input
    # channels
    if args.channels == 2 and not args.t1:
        parser.error("--channels 2 requires --t1.")
    if args.channels == 1 and args.t1:
        parser.error("--channels 1 does not use --t1; drop it or pass --channels 2.")

    preprocess_session(flair=args.flair,
                       gt_seg=args.ground_truth_seg,
                       output=args.output,
                       session_id=args.session_id,
                       stripped=args.stripped,
                       t1=args.t1,
                       fast=args.fast,
                       device=args.device,
                       threads=args.threads)
    print("Done.")
