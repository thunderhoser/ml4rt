"""Creates figure showing model evaluation by site."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
SITES_FOR_RELIA_ARG_NAME = 'site_names_for_reliability'
SITE_FOR_HISTO_ARG_NAME = 'site_name_for_histograms'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing evaluation figures created by '
    'plot_evaluation_by_time.py.  This script will panel some of those figures '
    'together.'
)
SITES_FOR_RELIA_HELP_STRING = (
    'Names of sites for which height-coded reliability curves will be plotted.'
    '  Site names must match those in file names, and there must be 3 sites.'
)
SITE_FOR_HISTO_HELP_STRING = (
    'Name of site used for inset histograms in attributes diagrams.  Site name '
    'must match those in file names.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Output images (paneled figure and temporary '
    'figures) will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SITES_FOR_RELIA_ARG_NAME, type=str, nargs=3, required=True,
    help=SITES_FOR_RELIA_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SITE_FOR_HISTO_ARG_NAME, type=str, required=True,
    help=SITE_FOR_HISTO_HELP_STRING
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


def _run(input_dir_name, site_names_for_reliability, site_name_for_histogram,
         output_dir_name):
    """Creates figure showing overall model evaluation.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param site_names_for_reliability: Same.
    :param site_name_for_histogram: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    pathless_input_file_names = [
        'shortwave-surface-down-flux-w-m02_attributes_{0:s}.jpg'.format(
            site_name_for_histogram
        ),
        'shortwave-toa-up-flux-w-m02_attributes_{0:s}.jpg'.format(
            site_name_for_histogram
        ),
        'net-shortwave-flux-w-m02_attributes_{0:s}.jpg'.format(
            site_name_for_histogram
        )
    ]

    num_sites_for_relia = len(site_names_for_reliability)

    for j in range(num_sites_for_relia):
        this_file_name = (
            'shortwave-heating-rate-k-day01_reliability_{0:s}.jpg'
        ).format(site_names_for_reliability[j])

        pathless_input_file_names.append(this_file_name)

    pathless_input_file_names += [
        'shortwave-heating-rate-k-day01_bias_profile.jpg',
        'shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
        'shortwave-heating-rate-k-day01_mae-skill-score_profile.jpg'
    ]

    panel_file_names = [
        '{0:s}/{1:s}'.format(input_dir_name, p)
        for p in pathless_input_file_names
    ]
    resized_panel_file_names = [
        '{0:s}/{1:s}'.format(output_dir_name, p)
        for p in pathless_input_file_names
    ]

    letter_label = None

    for i in range(len(panel_file_names)):
        print('Resizing panel and saving to: "{0:s}"...'.format(
            resized_panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=resized_panel_file_names[i]
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _overlay_text(
            image_file_name=resized_panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.resize_image(
            input_file_name=resized_panel_file_names[i],
            output_file_name=resized_panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/evaluation_by_site.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

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
        site_names_for_reliability=getattr(
            INPUT_ARG_OBJECT, SITES_FOR_RELIA_ARG_NAME
        ),
        site_name_for_histogram=getattr(
            INPUT_ARG_OBJECT, SITE_FOR_HISTO_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
