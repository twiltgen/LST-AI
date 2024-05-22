import subprocess
import shlex

def run_N4(input_image, output_image, bias_field):
    """
    Runs the ANTs N4BiasFieldCorrection tool to perform bias field correction on an input image.

    Parameters:
    input_image (str): Path to the input image file.
    output_image (str): Path for the output image file.
    bias_field (str): Path for the output bias field file.

    This function utilizes ANTs N4BiasFieldCorrection, a tool for bias field correction MRI images.
    """
    N4_call = f'N4BiasFieldCorrection -d 3 -i {input_image} -o [{output_image}, {bias_field}]'
    subprocess.run(shlex.split(N4_call), check=True)