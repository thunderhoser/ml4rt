"""Silly fuckery."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import imagemagick_utils

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

IMAGE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/experiment06/'
    'num-dense-layers=4_dense-dropout=0.300_scalar-lf-weight=001.0/'
    'model_epoch=205_val-loss=0.052551/isotonic_regression/'
    'validation_tropical_sites/evaluation'
)

PATHLESS_INPUT_FILE_NAMES = [
    'shortwave-heating-rate-k-day01_bias_profile.jpg',
    'shortwave-heating-rate-k-day01_reliability_new-model.jpg',
    'shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
    'shortwave-surface-down-flux-w-m02_attributes_new-model.jpg',
    'shortwave-heating-rate-k-day01_mae-skill-score_profile.jpg',
    'shortwave-toa-up-flux-w-m02_attributes_new-model.jpg',
    'shortwave-heating-rate-k-day01_correlation_profile.jpg',
    'net-shortwave-flux-w-m02_attributes_new-model.jpg',
]

LETTER_LABELS = ['(a)', '(e)', '(b)', '(f)', '(c)', '(g)', '(d)', '(h)']

OUTPUT_FILE_NAME = '{0:s}/jtti-rt_figure02.jpg'.format(IMAGE_DIR_NAME)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Overlays text on image.

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    print(command_string)

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


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

    _overlay_text(
        image_file_name=resized_file_names[i],
        x_offset_from_left_px=100, y_offset_from_top_px=100,
        text_string=LETTER_LABELS[i]
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
