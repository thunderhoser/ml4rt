"""Creates 10-panel figure showing evaluation metrics for one model.

This includes evaluation metrics for both deterministic predictions and
uncertainty quantification (UQ).
"""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils

INPUT_FILE_SUFFIXES_10PANELS = [
    'evaluation/net-shortwave-flux-w-m02_attributes_new-model.jpg',
    'spread_vs_skill/spread_vs_skill_net-shortwave-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_attributes_new-model.jpg',
    'discard_test_for_heating_rates/discard_test_net-shortwave-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
    'pit_histograms/pit_histogram_net-shortwave-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_bias_profile.jpg'
]

PANEL_LETTERS_10PANELS = ['a', 'e', 'h', 'b', 'f', 'i', 'c', 'g', 'j', 'd']

INPUT_FILE_SUFFIXES_12PANELS = [
    'evaluation/net-shortwave-flux-w-m02_attributes_new-model.jpg',
    'spread_vs_skill/spread_vs_skill_net-shortwave-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_attributes_new-model.jpg',
    'discard_test_for_heating_rates/discard_test_net-shortwave-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
    'pit_histograms/pit_histogram_net-shortwave-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_shortwave-heating-rate-k-day01.jpg',
    'evaluation/shortwave-heating-rate-k-day01_bias_profile.jpg',
    'spread_vs_skill/ssrat_shortwave-heating-rate-k-day01.jpg',
    'pit_histograms/extreme_pit_frequency_shortwave-heating-rate-k-day01.jpg'
]

PANEL_LETTERS_12PANELS = [
    'a', 'e', 'i', 'b', 'f', 'j', 'c', 'g', 'k', 'd', 'h', 'l'
]

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
MAKE_12PANEL_FIG_ARG_NAME = 'make_12panel_figure'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory -- containing evaluation figures created by '
    'plot_evaluation.py, plot_spread_vs_skill.py, plot_discard_test.py, and '
    'plot_pit_histograms.py.  This script will panel some of '
    'those figures together.'
)
MAKE_12PANEL_FIG_HELP_STRING = (
    'Boolean flag.  If 1, will make 12-panel figure, including SSRAT and '
    'extreme-PIT-frequency profiles for heating rate.  If 0, will make '
    '10-panel figure.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  New images (paneled figures) will be saved '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAKE_12PANEL_FIG_ARG_NAME, type=int, required=False, default=0,
    help=MAKE_12PANEL_FIG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

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


def _run(input_dir_name, make_12panel_figure, output_dir_name):
    """Creates 10-panel figure showing evaluation metrics for one model.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param make_12panel_figure: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    concat_figure_file_name = '{0:s}/overall_eval_determ_and_uq.jpg'.format(
        output_dir_name
    )

    if make_12panel_figure:
        input_file_suffixes = INPUT_FILE_SUFFIXES_12PANELS
        panel_letters = PANEL_LETTERS_12PANELS
    else:
        input_file_suffixes = INPUT_FILE_SUFFIXES_10PANELS
        panel_letters = PANEL_LETTERS_10PANELS

    panel_file_names = [
        '{0:s}/{1:s}'.format(input_dir_name, p) for p in input_file_suffixes
    ]
    resized_panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, p.split('/')[-1])
        for p in input_file_suffixes
    ]

    for i in range(len(panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            resized_panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=resized_panel_file_names[i]
        )

        _overlay_text(
            image_file_name=resized_panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(panel_letters[i])
        )
        imagemagick_utils.resize_image(
            input_file_name=resized_panel_file_names[i],
            output_file_name=resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    print('Concatenating panels to: "{0:s}"...'.format(
        concat_figure_file_name
    ))

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
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        make_12panel_figure=bool(
            getattr(INPUT_ARG_OBJECT, MAKE_12PANEL_FIG_ARG_NAME)
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
