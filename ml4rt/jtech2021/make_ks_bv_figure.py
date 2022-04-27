"""Creates figure showing Kolmogorov-Smirnov and bias-variance results."""

import os
import argparse
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils

PATHLESS_INPUT_FILE_NAMES = [
    'shortwave-heating-rate-k-day01_mse-bias_profile.jpg',
    'shortwave-heating-rate-k-day01_mse-variance_profile.jpg',
    'shortwave-heating-rate-k-day01_ks-statistic_profile.jpg',
    'shortwave-heating-rate-k-day01_ks-p-value_profile.jpg'
]

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 2
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

OVERALL_EVAL_DIR_ARG_NAME = 'input_overall_eval_dir_name'
BY_CLOUD_EVAL_DIR_ARG_NAME = 'input_by_cloud_eval_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

OVERALL_EVAL_DIR_HELP_STRING = (
    'Name of directory with evaluation figures (created by plot_evaluation.py) '
    'for overall dataset.'
)
BY_CLOUD_EVAL_DIR_HELP_STRING = (
    'Name of directory with evaluation figures (created by plot_evaluation.py) '
    'by cloud regime.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Output images (paneled figure and temporary '
    'figures) will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OVERALL_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=OVERALL_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BY_CLOUD_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=BY_CLOUD_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


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

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _run(overall_eval_dir_name, by_cloud_eval_dir_name, output_dir_name):
    """Creates figure showing Kolmogorov-Smirnov and bias-variance results.

    This is effectively the main method.

    :param overall_eval_dir_name: See documentation at top of file.
    :param by_cloud_eval_dir_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    first_panel_file_names = [
        '{0:s}/{1:s}'.format(overall_eval_dir_name, p)
        for p in PATHLESS_INPUT_FILE_NAMES
    ]
    first_resized_panel_file_names = [
        '{0:s}/overall_{1:s}'.format(output_dir_name, p)
        for p in PATHLESS_INPUT_FILE_NAMES
    ]

    second_panel_file_names = [
        '{0:s}/{1:s}'.format(by_cloud_eval_dir_name, p)
        for p in PATHLESS_INPUT_FILE_NAMES
    ]
    second_resized_panel_file_names = [
        '{0:s}/by_cloud_{1:s}'.format(output_dir_name, p)
        for p in PATHLESS_INPUT_FILE_NAMES
    ]

    letter_label = None

    for i in range(len(first_panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            first_resized_panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=first_panel_file_names[i],
            output_file_name=first_resized_panel_file_names[i]
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _overlay_text(
            image_file_name=first_resized_panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.resize_image(
            input_file_name=first_resized_panel_file_names[i],
            output_file_name=first_resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    for i in range(len(second_panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            second_resized_panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=second_panel_file_names[i],
            output_file_name=second_resized_panel_file_names[i]
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _overlay_text(
            image_file_name=second_resized_panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.resize_image(
            input_file_name=second_resized_panel_file_names[i],
            output_file_name=second_resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/overall_evaluation.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    resized_panel_file_names = []
    for i in range(len(first_panel_file_names)):
        resized_panel_file_names.append(first_resized_panel_file_names[i])
        resized_panel_file_names.append(second_resized_panel_file_names[i])

    imagemagick_utils.concatenate_images(
        input_file_names=resized_panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        overall_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, OVERALL_EVAL_DIR_ARG_NAME
        ),
        by_cloud_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, BY_CLOUD_EVAL_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
