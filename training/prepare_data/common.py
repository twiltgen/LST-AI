#!/usr/bin/env python3
# coding: utf-8

"""Preprocessing shared by the training and inference data preparation.

Both pipelines under prepare_data/ must register, strip and warp identically --
a model has to be served at inference what it was trained on -- so the work
itself lives here once and training/ and inference/ only differ in what they
require and what they write:

    training/    the ground truth is required; the affine is not kept.
    inference/   the ground truth is optional; the FLAIR affine is kept so a
                 prediction can be warped back to native FLAIR space.

Nothing here is meant to be run directly; use the prepare_cohort.py or
preprocess_session.py of whichever pipeline you want.

REQUIRES
> greedy (picsl_greedy)
> HD-BET
> MNI atlas files (downloaded on first run)
"""
import os
import multiprocessing
import glob
import tempfile
import shutil
from collections import namedtuple

import nibabel as nib
import numpy as np

from lst_ai.strip import run_hdbet, apply_mask
from lst_ai.register import mni_registration, rigid_reg, apply_warp_label
from lst_ai.utils import DATA_DIR, download_data, harmonize_affines

CITATION = (
    "###########################\n\n"
    "Thank you for using LST-AI. If you publish your results, please cite our paper:\n"
    "Wiltgen T, McGinnis J, Schlaeger S, Kofler F, Voon C, Berthele A, Bischl D, Grundl L, Will N, Metz M, Schinz D, "
    "Sepp D, Prucker P, Schmitz-Koep B, Zimmer C, Menze B, Rueckert D, Hemmer B, Kirschke J, Mühlau M, Wiestler B. "
    "LST-AI: A Deep Learning Ensemble for Accurate MS Lesion Segmentation. NeuroImage: Clinical, Volume 42, 2024. "
    "https://doi.org/10.1016/j.nicl.2024.103611.\n"
    "###########################\n"
)

# One discovered session. ``mask`` is the path the mask was looked for at, which
# exists only when ``has_mask``; ``missing`` lists the required files that are absent.
Session = namedtuple('Session', 'session_id flair t1 mask has_mask missing')


def resolve_atlas(stripped):
    """
    Returns the MNI atlas to register against, downloading the atlas files on first use.

    Parameters
    ----------
    stripped : bool
        If True, returns the skull-stripped atlas, which is what already skull-stripped
        inputs must be registered to. If False, returns the full-head atlas.

    Returns
    -------
    str
        Path to the atlas NIfTI file.
    """
    print(f"Looking for atlas files in {DATA_DIR}.")
    download_data(path=DATA_DIR)
    name = "sub-mni152_space-mni_t1bet.nii.gz" if stripped else "sub-mni152_space-mni_t1.nii.gz"
    return os.path.join(DATA_DIR, "atlas", name)


def register_strip_2channel(t1, flair, work_dir, stem, stripped, atlas, fast, device, threads):
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

    if stripped:
        shutil.copy(mni_t1w, stripped_t1w)
        shutil.copy(mni_flair, stripped_flair)
    else:
        run_hdbet(input_image=mni_t1w,
                  output_image=stripped_t1w,
                  device=device,
                  mode="fast" if fast else "accurate")
        shutil.move(stripped_t1w.replace(".nii.gz", "_bet.nii.gz"), brainmask)

        apply_mask(input_image=mni_flair,
                   mask=brainmask,
                   output_image=stripped_flair)
    return stripped_t1w, stripped_flair, affine_flair


def register_strip_1channel(flair, work_dir, stem, stripped, atlas, fast, device, threads):
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

    if stripped:
        shutil.copy(mni_flair, stripped_flair)
    else:
        run_hdbet(input_image=mni_flair,
                  output_image=stripped_flair,
                  device=device,
                  mode="fast" if fast else "accurate")
        shutil.move(stripped_flair.replace(".nii.gz", "_bet.nii.gz"), brainmask)

    return None, stripped_flair, affine_flair


def preprocess_session(flair, output, session_id, gt_seg=None, stripped=False, t1=None,
                       fast=False, device='0', threads=None, save_affine=False):
    """Preprocess one session. It deals with both 1-channel and 2-channel modes, depending on whether ``t1`` is None.
    First, it harmonizes the affines of the input images, then registers them to MNI space,
    skull-strips them, and warps the ground truth segmentation to match the FLAIR in MNI space.

    Parameters
    ----------
    flair : str
        Path to the FLAIR image (zipped nifti).
    output : str
        Path to the output directory where the processed images will be saved.
    session_id : str
        Session identifier used as the stem of the output filenames.
    gt_seg : str, optional
        Path to the ground truth segmentation in native FLAIR space (zipped nifti).
        Default is None, which skips it: inference needs no ground truth.
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
    save_affine : bool, optional
        If True, also writes the FLAIR-to-MNI affine, which warps a prediction back to
        native FLAIR space. Default is False.

    Returns
    -------
    dict
        A dictionary containing the paths to the processed images:
        - 't1': Path to the processed T1w image (None in 1-channel mode).
        - 'flair': Path to the processed FLAIR image.
        - 'seg': Path to the processed ground truth segmentation (None when no ground truth was given).
        - 'affine': Path to the FLAIR-to-MNI affine (None unless ``save_affine``).
    """
    threads = threads or multiprocessing.cpu_count()

    # Check input files and output directory
    assert os.path.exists(flair), 'LST.AI aborted. FLAIR Image Path does not exist.'
    assert str(flair).endswith(".nii.gz"), 'Please provide FLAIR as a zipped nifti.'
    if gt_seg is not None:
        assert os.path.exists(gt_seg), 'LST.AI aborted. Ground Truth Segmentation Path does not exist.'
        assert str(gt_seg).endswith(".nii.gz"), 'Please provide Ground Truth Segmentation as a zipped nifti.'
    if t1 is not None:
        assert os.path.exists(t1), 'LST.AI aborted. T1w Image Path does not exist.'
        assert str(t1).endswith(".nii.gz"), 'Please provide T1w as a zipped nifti.'
    assert not os.path.isfile(output), 'Please provide an output path, not a filename.'
    assert session_id and os.sep not in session_id, 'Please provide a session id (a filename stem, not a path).'

    atlas = resolve_atlas(stripped)

    work_dir = tempfile.mkdtemp(prefix='lst_ai_')

    try:
        # Create a session directory in the output path
        session_dir = os.path.abspath(os.path.join(output, session_id))
        os.makedirs(session_dir, exist_ok=True)

        # harmonize sform and qform of the input images to avoid registration issues
        harmonized_flair = os.path.join(work_dir, 'input_FLAIR.nii.gz')
        harmonize_affines(flair, harmonized_flair)
        harmonized_seg = None
        if gt_seg is not None:
            harmonized_seg = os.path.join(work_dir, 'input_GT_seg.nii.gz')
            harmonize_affines(gt_seg, harmonized_seg)
        harmonized_t1 = None
        if t1 is not None:
            harmonized_t1 = os.path.join(work_dir, 'input_T1w.nii.gz')
            harmonize_affines(t1, harmonized_t1)

        # run registration and skull-stripping, depending on the number of channels
        print(f"{session_id}: Images are registered to MNI and skull-stripped...")
        if t1 is not None:
            stripped_t1w, stripped_flair, affine_flair = register_strip_2channel(t1=harmonized_t1,
                                                                                 flair=harmonized_flair,
                                                                                 work_dir=work_dir,
                                                                                 stem=session_id,
                                                                                 stripped=stripped,
                                                                                 atlas=atlas,
                                                                                 fast=fast,
                                                                                 device=device,
                                                                                 threads=threads)
        else:
            stripped_t1w, stripped_flair, affine_flair = register_strip_1channel(flair=harmonized_flair,
                                                                                 work_dir=work_dir,
                                                                                 stem=session_id,
                                                                                 stripped=stripped,
                                                                                 atlas=atlas,
                                                                                 fast=fast,
                                                                                 device=device,
                                                                                 threads=threads)

        written = {'t1': None,
                   'flair': os.path.join(session_dir, f"{session_id}_flair.nii.gz"),
                   'seg': None,
                   'affine': None}

        # The ground truth is in native FLAIR space, so it needs to be registered using the FLAIR affine.
        if harmonized_seg is not None:
            print(f"{session_id}: Ground truth segmentation is registered to MNI space...")
            mni_seg = os.path.join(work_dir, f'{session_id}_space-mni_seg-GT.nii.gz')
            apply_warp_label(image_org_space=stripped_flair,
                             affine=affine_flair,
                             origin=harmonized_seg,
                             target=mni_seg,
                             reverse=False,
                             n_threads=threads)
            written['seg'] = os.path.join(session_dir, f"{session_id}_seg.nii.gz")

        print(f"{session_id}: Copy files to output directory...")
        shutil.copy(stripped_flair, written['flair'])

        if written['seg'] is not None:
            shutil.copy(mni_seg, written['seg'])

        if stripped_t1w is not None:
            written['t1'] = os.path.join(session_dir, f"{session_id}_t1.nii.gz")
            shutil.copy(stripped_t1w, written['t1'])

        if save_affine:
            written['affine'] = os.path.join(session_dir, f"{session_id}_flair-to-mni.mat")
            shutil.copy(affine_flair, written['affine'])

        print(f"{session_id}: Results in {session_dir}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return written


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


def find_sessions(bids_root, mask_root, flair_suffix, t1_suffix, mask_suffix, channels,
                  require_mask):
    """
    Finds all sessions in the BIDS dataset by globbing for FLAIR files with the specified suffix.
    For each FLAIR file found, it derives the corresponding T1w and lesion mask paths based on the session prefix
    and checks for their existence.

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
    require_mask : bool
        If True, a session without a lesion mask counts as incomplete, which is what training
        needs. If False, a missing mask is not an error and the session is prepared without a
        ground truth.

    Returns
    -------
    list of Session
        One Session per FLAIR found, each carrying:
        - session_id (str): The session identifier (e.g., 'sub-01_ses-01').
        - flair (str): The path to the FLAIR image.
        - t1 (str or None): The path to the T1w image, or None if not used.
        - mask (str): The path the lesion mask was looked for at, which only exists when has_mask.
        - has_mask (bool): Whether that lesion mask was found.
        - missing (list of str): The required files that are absent. Empty when the session is complete.
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

        has_mask = os.path.exists(mask)
        missing = [mask] if require_mask and not has_mask else []
        missing += [p for p in ([t1] if t1 else []) if not os.path.exists(p)]
        sessions.append(Session(session_id, flair, t1, mask, has_mask, missing))
    return sessions
