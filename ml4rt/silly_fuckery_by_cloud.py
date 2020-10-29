"""Silly fuckery."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import imagemagick_utils

IMAGE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/experiment06/'
    'num-dense-layers=4_dense-dropout=0.300_scalar-lf-weight=001.0/'
    'model_epoch=205_val-loss=0.052551/isotonic_regression/'
    'validation_tropical_sites/by_cloud_regime/evaluation'
)

PATHLESS_INPUT_FILE_NAMES = [
    'shortwave-heating-rate-k-day01_bias_profile.jpg',
    'shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
    'shortwave-heating-rate-k-day01_reliability_no-cloud.jpg',
    'shortwave-surface-down-flux-w-m02_attributes_no-cloud.jpg',
    'shortwave-heating-rate-k-day01_reliability_single-layer-cloud.jpg',
    'shortwave-toa-up-flux-w-m02_attributes_no-cloud.jpg',
    'shortwave-heating-rate-k-day01_reliability_multi-layer-cloud.jpg',
    'net-shortwave-flux-w-m02_attributes_no-cloud.jpg'
]

OUTPUT_FILE_NAME = '{0:s}/jtti-rt_figure03.jpg'.format(IMAGE_DIR_NAME)

input_file_names = [
    '{0:s}/{1:s}'.format(IMAGE_DIR_NAME, f) for f in PATHLESS_INPUT_FILE_NAMES
]
resized_file_names = [
    f.replace('.jpg', '_resized.jpg') for f in input_file_names
]

for i in range(len(input_file_names)):
    print(input_file_names[i])

    imagemagick_utils.trim_whitespace(
        input_file_name=input_file_names[i],
        output_file_name=resized_file_names[i]
    )

    imagemagick_utils.resize_image(
        input_file_name=input_file_names[i],
        output_file_name=resized_file_names[i],
        output_size_pixels=int(2.5e6)
    )

imagemagick_utils.concatenate_images(
    input_file_names=input_file_names, output_file_name=OUTPUT_FILE_NAME,
    num_panel_rows=4, num_panel_columns=2
)