#!/usr/bin/env python -W ignore::DeprecationWarning
# coding: utf-8

"""
REQUIRES
> greedy
> HD-BET
> MNI atlas files
> Model files
"""
import os
import multiprocessing
import tempfile
import shutil
import argparse

from lst_ai.strip import run_hdbet, apply_mask
from lst_ai.register import mni_registration, apply_warp_label
from lst_ai.utils import DATA_DIR, download_data, harmonize_affines

if __name__ == "__main__":

    print("###########################\n")
    print("Thank you for using LST-AI. If you publish your results, please cite our paper:")
    print("Wiltgen T, McGinnis J, Schlaeger S, Kofler F, Voon C, Berthele A, Bischl D, Grundl L, Will N, Metz M, Schinz D, "
          "Sepp D, Prucker P, Schmitz-Koep B, Zimmer C, Menze B, Rueckert D, Hemmer B, Kirschke J, Mühlau M, Wiestler B. "
          "LST-AI: A Deep Learning Ensemble for Accurate MS Lesion Segmentation. NeuroImage: Clinical, Volume 42, 2024. "
          "https://doi.org/10.1016/j.nicl.2024.103611.")
    print("###########################\n")

    parser = argparse.ArgumentParser(description='Prepare data to train a new LST-AI model.')

    # Input Images
    parser.add_argument('--t1',
                        dest='t1',
                        help='Path to T1 image',
                        type=str,
                        required=True)
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
                        help='Path to output directory for the MNI-space T1w, FLAIR and lesion mask.',
                        type=str,
                        required=True)

    # Names the three output files. lst_training.data.MSDataset globs for
    # <subject_id>_flair.nii.gz.
    parser.add_argument('--subject_id',
                        dest='subject_id',
                        help='Subject identifier used as the stem of the output filenames.',
                        type=str,
                        required=True)

    # Temporary directory
    parser.add_argument('--temp',
                        dest='temp',
                        default='',
                        help='Path to temp directory.',
                        type=str)

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

    print(f"Looking for model weights in {DATA_DIR}.")
    download_data(path=DATA_DIR)

    # Sanity Checks
    assert os.path.exists(args.t1), 'LST.AI aborted. T1w Image Path does not exist.'
    assert os.path.exists(args.flair), 'LST.AI aborted. Flair Image Path does not exist.'
    assert os.path.exists(args.ground_truth_seg), 'LST.AI aborted. Ground Truth Segmentation Path does not exist.'
    assert str(args.t1).endswith(".nii.gz"), 'Please provide T1w as a zipped nifti.'
    assert str(args.flair).endswith(".nii.gz"), 'Please provide FLAIR as a zipped nifti.'
    assert str(args.ground_truth_seg).endswith(".nii.gz"), 'Please provide Ground Truth Segmentation as a zipped nifti.'
    assert not os.path.isfile(args.output), 'Please provide an output path, not a filename.'
    assert args.subject_id and os.sep not in args.subject_id, \
        'Please provide a subject id (a filename stem, not a path).'

    if not args.temp:
        work_dir = tempfile.mkdtemp(prefix='lst_ai_')
    else:
        work_dir = os.path.abspath(args.temp)
        # make temp directory in case it does not exist
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    # Harmonize the qform and sform of the inputs before anything reads them.
    # nibabel (used throughout this pipeline) prefers the sform, while greedy
    # reads the qform; if a prior co-registration updated only the sform, the two
    # disagree and the segmentation is mislocated. See CompImg/LST-AI#44. The
    # originals are left untouched; the pipeline continues on work-dir copies.
    harmonized_t1 = os.path.join(work_dir, 'input_T1w.nii.gz')
    harmonized_flair = os.path.join(work_dir, 'input_FLAIR.nii.gz')
    harmonize_affines(args.t1, harmonized_t1)
    harmonize_affines(args.flair, harmonized_flair)
    args.t1 = harmonized_t1
    args.flair = harmonized_flair
    # An existing segmentation is read in FLAIR space by the same tools, so it
    # has to be harmonized the same way or it would disagree with the harmonized
    # FLAIR exactly as the raw inputs disagreed with each other.
    harmonized_seg = os.path.join(work_dir, 'input_GT_seg.nii.gz')
    harmonize_affines(args.ground_truth_seg, harmonized_seg)
    args.ground_truth_seg = harmonized_seg

    #  Define Image Paths (original space)
    path_orig_t1w = os.path.join(work_dir, 'sub-X_ses-Y_space-t1w_T1w.nii.gz')
    path_orig_flair = os.path.join(work_dir, 'sub-X_ses-Y_space-flair_FLAIR.nii.gz')

    #  Define Image Paths (MNI space)
    path_mni_t1w = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_T1w.nii.gz')
    path_mni_flair = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_FLAIR.nii.gz')
    path_mni_stripped_t1w = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_desc-stripped_T1w.nii.gz')
    path_mni_stripped_flair = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_desc-stripped_FLAIR.nii.gz')

    # Masks
    path_mni_brainmask = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_brainmask.nii.gz')

    # Temp Segmentation results
    path_flair_segmentation =  os.path.join(work_dir, 'sub-X_ses-Y_space-flair_seg-GT.nii.gz')
    path_mni_segmentation = os.path.join(work_dir, 'sub-X_ses-Y_space-mni_seg-GT.nii.gz')

    # Output paths (in MNI space), named for lst_training.data.MSDataset
    filename_output_t1w = f"{args.subject_id}_t1.nii.gz"
    filename_output_flair = f"{args.subject_id}_flair.nii.gz"
    filename_output_segmentation = f"{args.subject_id}_seg.nii.gz"

    # affines
    path_affine_mni_t1w = os.path.join(work_dir, 'affine_t1w_to_mni.mat')
    path_affine_mni_flair = os.path.join(work_dir, 'affine_flair_to_mni.mat')

    # atlas files
    t1w_atlas = os.path.join(DATA_DIR, "atlas", "sub-mni152_space-mni_t1.nii.gz")

    # make output path (in case it does not exist)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # start preprocessing
    print("Images need to be registered to MNI. Processing with Greedy.")
    shutil.copy(args.t1, path_orig_t1w)
    shutil.copy(args.flair, path_orig_flair)
    shutil.copy(args.ground_truth_seg, path_flair_segmentation)


    ## register images to MNI space
    # first register T1w and FLAIR to MNI space
    mni_registration(t1w_atlas,
                     path_orig_t1w,
                     path_orig_flair,
                     path_mni_t1w,
                     path_mni_flair,
                     path_affine_mni_t1w,
                     path_affine_mni_flair,
                     n_threads=args.threads)


    ## apply skull stripping to T1w and FLAIR images
    # run HD-BET on T1w image
    if args.fast:
        run_hdbet(input_image=path_mni_t1w, output_image=path_mni_stripped_t1w, device=args.device, mode="fast")
    else:
        run_hdbet(input_image=path_mni_t1w, output_image=path_mni_stripped_t1w, device=args.device, mode="accurate")

    # move processed mask to correct naming convention
    hdbet_mask = path_mni_stripped_t1w.replace(".nii.gz", "_bet.nii.gz")
    shutil.move(hdbet_mask, path_mni_brainmask)

    # then apply brain mask to FLAIR
    apply_mask(input_image=path_mni_flair,
                mask=path_mni_brainmask,
                output_image=path_mni_stripped_flair)

    shutil.copy(path_mni_stripped_t1w, os.path.join(args.output, filename_output_t1w))
    shutil.copy(path_mni_stripped_flair, os.path.join(args.output, filename_output_flair))


    ## register the ground truth segmentation to MNI space
    # warp segmentation to MNI space using the FLAIR transformation
    apply_warp_label(image_org_space=path_mni_stripped_flair,
                     affine=path_affine_mni_flair,
                     target=path_mni_segmentation,
                     origin=path_flair_segmentation,
                     reverse=False,
                     n_threads=args.threads)

    # store the segmentations
    shutil.copy(path_mni_segmentation, os.path.join(args.output, filename_output_segmentation))

    print(f"Results in {work_dir}")
    if not args.temp:
        print(f"Delete temporary directory: {work_dir}")
        shutil.rmtree(work_dir)
    print("Done.")
