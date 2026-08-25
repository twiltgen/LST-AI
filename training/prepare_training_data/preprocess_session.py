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

REQUIRES
> greedy (picsl_greedy)
> HD-BET
> MNI atlas files (downloaded on first run)
"""
import os
import multiprocessing
import tempfile
import shutil
import argparse

from lst_ai.strip import run_hdbet, apply_mask
from lst_ai.register import mni_registration, rigid_reg, apply_warp_label
from lst_ai.utils import DATA_DIR, download_data, harmonize_affines


def _register_2channel(t1, flair, work_dir, stem, atlas, fast, device, threads):
    """T1w -> atlas, FLAIR -> MNI T1w; HD-BET on the T1w, its mask applied to the FLAIR."""
    mni_t1w = os.path.join(work_dir, f'{stem}_space-mni_T1w.nii.gz')
    mni_flair = os.path.join(work_dir, f'{stem}_space-mni_FLAIR.nii.gz')
    stripped_t1w = os.path.join(work_dir, f'{stem}_space-mni_desc-stripped_T1w.nii.gz')
    stripped_flair = os.path.join(work_dir, f'{stem}_space-mni_desc-stripped_FLAIR.nii.gz')
    brainmask = os.path.join(work_dir, f'{stem}_space-mni_brainmask.nii.gz')
    affine_t1w = os.path.join(work_dir, 'affine_t1w_to_mni.mat')
    affine_flair = os.path.join(work_dir, 'affine_flair_to_mni.mat')

    mni_registration(atlas_t1=atlas, 
                     path_org_t1=t1, 
                     path_org_flair=flair, 
                     path_mni_t1=mni_t1w, 
                     path_mni_flair=mni_flair,
                     path_t1_affine=affine_t1w, 
                     path_flair_affine=affine_flair, 
                     n_threads=threads)

    run_hdbet(input_image=mni_t1w, 
              output_image=stripped_t1w,
              device=device, 
              mode="fast" if fast else "accurate")
    shutil.move(stripped_t1w.replace(".nii.gz", "_bet.nii.gz"), brainmask)

    apply_mask(input_image=mni_flair, 
               mask=brainmask, 
               output_image=stripped_flair)
    return stripped_t1w, stripped_flair, affine_flair


def _register_1channel(flair, work_dir, stem, atlas, fast, device, threads):
    """FLAIR -> atlas directly, HD-BET on the FLAIR. No T1w involved."""
    mni_flair = os.path.join(work_dir, f'{stem}_space-mni_FLAIR.nii.gz')
    stripped_flair = os.path.join(work_dir, f'{stem}_space-mni_desc-stripped_FLAIR.nii.gz')
    brainmask = os.path.join(work_dir, f'{stem}_space-mni_brainmask.nii.gz')
    affine_flair = os.path.join(work_dir, 'affine_flair_to_mni.mat')

    rigid_reg(moving=flair, 
              fixed=atlas, 
              affine=affine_flair,
              destination=mni_flair, 
              n_threads=threads)

    run_hdbet(input_image=mni_flair, 
              output_image=stripped_flair,
              device=device, 
              mode="fast" if fast else "accurate")
    shutil.move(stripped_flair.replace(".nii.gz", "_bet.nii.gz"), brainmask)

    return None, stripped_flair, affine_flair


def preprocess_session(flair, gt_seg, output, session_id, t1=None,
                       fast=False, device='0', threads=None):
    """Preprocess one session. It deals with both 1-channel and 2-channel modes, depending on whether ``t1`` is None.
    First, it harmonizes the affines of the input images, then registers them to MNI space, 
    skull-strips them, and warps the ground truth segmentation to match the FLAIR in MNI space.

    Parameters
    ----------
    flair : str
        Path to the FLAIR image (zipped nifti).
    gt_seg : str
        Path to the ground truth segmentation (zipped nifti).
    output : str
        Path to the output directory where the processed images will be saved.
    session_id : str
        Session identifier used as the stem of the output filenames.
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
    threads = threads or multiprocessing.cpu_count()
    
    # Check input files and output directory
    assert os.path.exists(flair), 'LST.AI aborted. FLAIR Image Path does not exist.'
    assert os.path.exists(gt_seg), 'LST.AI aborted. Ground Truth Segmentation Path does not exist.'
    assert str(flair).endswith(".nii.gz"), 'Please provide FLAIR as a zipped nifti.'
    assert str(gt_seg).endswith(".nii.gz"), 'Please provide Ground Truth Segmentation as a zipped nifti.'
    if t1 is not None:
        assert os.path.exists(t1), 'LST.AI aborted. T1w Image Path does not exist.'
        assert str(t1).endswith(".nii.gz"), 'Please provide T1w as a zipped nifti.'
    assert not os.path.isfile(output), 'Please provide an output path, not a filename.'
    assert session_id and os.sep not in session_id, 'Please provide a session id (a filename stem, not a path).'

    print(f"Looking for atlas files in {DATA_DIR}.")
    download_data(path=DATA_DIR)
    atlas = os.path.join(DATA_DIR, "atlas", "sub-mni152_space-mni_t1.nii.gz")

    work_dir = tempfile.mkdtemp(prefix='lst_ai_')

    try:
        # Create a session directory in the output path
        session_dir = os.path.abspath(os.path.join(output, session_id))
        os.makedirs(session_dir, exist_ok=True)

        # harmonize sform and qform of the input images to avoid registration issues
        harmonized_flair = os.path.join(work_dir, 'input_FLAIR.nii.gz')
        harmonized_seg = os.path.join(work_dir, 'input_GT_seg.nii.gz')
        harmonize_affines(flair, harmonized_flair)
        harmonize_affines(gt_seg, harmonized_seg)
        harmonized_t1 = None
        if t1 is not None:
            harmonized_t1 = os.path.join(work_dir, 'input_T1w.nii.gz')
            harmonize_affines(t1, harmonized_t1)

        # run registration and skull-stripping, depending on the number of channels
        print("Images are registered to MNI. Processing with Greedy.")
        if harmonized_t1 is not None:
            stripped_t1w, stripped_flair, affine_flair = _register_2channel(t1=harmonized_t1, 
                                                                            flair=harmonized_flair, 
                                                                            work_dir=work_dir, 
                                                                            stem=session_id, 
                                                                            atlas=atlas, 
                                                                            fast=fast, 
                                                                            device=device, 
                                                                            threads=threads)
        else:
            stripped_t1w, stripped_flair, affine_flair = _register_1channel(flair=harmonized_flair, 
                                                                            work_dir=work_dir, 
                                                                            stem=session_id, 
                                                                            atlas=atlas, 
                                                                            fast=fast, 
                                                                            device=device, 
                                                                            threads=threads)

        # The ground truth is in native FLAIR space, so it needs to be registered using the FLAIR affine.
        mni_seg = os.path.join(work_dir, f'{session_id}_space-mni_seg-GT.nii.gz')
        apply_warp_label(image_org_space=stripped_flair,
                        affine=affine_flair,
                        origin=harmonized_seg,
                        target=mni_seg,
                        reverse=False,
                        n_threads=threads)

        written = {'t1': None,
                'flair': os.path.join(session_dir, f"{session_id}_flair.nii.gz"),
                'seg': os.path.join(session_dir, f"{session_id}_seg.nii.gz")}
        
        shutil.copy(stripped_flair, written['flair'])
        shutil.copy(mni_seg, written['seg'])

        if stripped_t1w is not None:
            written['t1'] = os.path.join(session_dir, f"{session_id}_t1.nii.gz")
            shutil.copy(stripped_t1w, written['t1'])

        print(f"Results in {session_dir}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return written


if __name__ == "__main__":

    print("###########################\n")
    print("Thank you for using LST-AI. If you publish your results, please cite our paper:")
    print("Wiltgen T, McGinnis J, Schlaeger S, Kofler F, Voon C, Berthele A, Bischl D, Grundl L, Will N, Metz M, Schinz D, "
          "Sepp D, Prucker P, Schmitz-Koep B, Zimmer C, Menze B, Rueckert D, Hemmer B, Kirschke J, Mühlau M, Wiestler B. "
          "LST-AI: A Deep Learning Ensemble for Accurate MS Lesion Segmentation. NeuroImage: Clinical, Volume 42, 2024. "
          "https://doi.org/10.1016/j.nicl.2024.103611.")
    print("###########################\n")

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

    if args.channels == 2 and not args.t1:
        parser.error("--channels 2 requires --t1.")
    if args.channels == 1 and args.t1:
        parser.error("--channels 1 does not use --t1; drop it or pass --channels 2.")

    preprocess_session(flair=args.flair,
                       gt_seg=args.ground_truth_seg,
                       output=args.output,
                       session_id=args.session_id,
                       t1=args.t1,
                       fast=args.fast,
                       device=args.device,
                       threads=args.threads)
    print("Done.")
