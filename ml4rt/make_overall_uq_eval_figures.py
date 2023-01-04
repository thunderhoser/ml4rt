"""Creates two figures showing overall evaluation of uncertainty quant (UQ)."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils

SHORTWAVE_FLUX_INPUT_FILE_SUFFIXES = [
    'spread_vs_skill/spread_vs_skill_shortwave-surface-down-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_shortwave-toa-up-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_net-shortwave-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_shortwave-surface-down-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_shortwave-toa-up-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_net-shortwave-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_shortwave-surface-down-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_shortwave-toa-up-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_net-shortwave-flux-w-m02.jpg'
]

LONGWAVE_FLUX_INPUT_FILE_SUFFIXES = [
    'spread_vs_skill/spread_vs_skill_longwave-surface-down-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_longwave-toa-up-flux-w-m02.jpg',
    'spread_vs_skill/spread_vs_skill_net-longwave-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_longwave-surface-down-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_longwave-toa-up-flux-w-m02.jpg',
    'discard_test_for_heating_rates/discard_test_net-longwave-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_longwave-surface-down-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_longwave-toa-up-flux-w-m02.jpg',
    'pit_histograms/pit_histogram_net-longwave-flux-w-m02.jpg'
]

SHORTWAVE_HR_INPUT_FILE_SUFFIXES = [
    'spread_vs_skill/spread_vs_skill_shortwave-heating-rate-k-day01.jpg',
    'discard_test_for_heating_rates/discard_test_shortwave-heating-rate-k-day01.jpg',
    'pit_histograms/pit_histogram_shortwave-heating-rate-k-day01.jpg',
    'spread_vs_skill/ssrel_shortwave-heating-rate-k-day01.jpg',
    'spread_vs_skill/ssrat_shortwave-heating-rate-k-day01.jpg',
    'discard_test_for_heating_rates/mono_fraction_shortwave-heating-rate-k-day01.jpg',
    'pit_histograms/pitd_shortwave-heating-rate-k-day01.jpg',
    'crps_plots/crps_shortwave-heating-rate-k-day01.jpg',
    'crps_plots/crpss_shortwave-heating-rate-k-day01.jpg'
]

LONGWAVE_HR_INPUT_FILE_SUFFIXES = [
    'spread_vs_skill/spread_vs_skill_longwave-heating-rate-k-day01.jpg',
    'discard_test_for_heating_rates/discard_test_longwave-heating-rate-k-day01.jpg',
    'pit_histograms/pit_histogram_longwave-heating-rate-k-day01.jpg',
    'spread_vs_skill/ssrel_longwave-heating-rate-k-day01.jpg',
    'spread_vs_skill/ssrat_longwave-heating-rate-k-day01.jpg',
    'discard_test_for_heating_rates/mono_fraction_longwave-heating-rate-k-day01.jpg',
    'pit_histograms/pitd_longwave-heating-rate-k-day01.jpg',
    'crps_plots/crps_longwave-heating-rate-k-day01.jpg',
    'crps_plots/crpss_longwave-heating-rate-k-day01.jpg'
]

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
FOR_SHORTWAVE_ARG_NAME = 'for_shortwave'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing evaluation figures created by '
    'plot_spread_vs_skill.py and plot_discard_test.py and '
    'plot_pit_histograms.py and plot_crps.py.  This script will panel some of '
    'those figures together.'
)
FOR_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will make figure with shortwave (longwave) '
    'errors.'
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
    '--' + FOR_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=FOR_SHORTWAVE_HELP_STRING
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


def _run(input_dir_name, for_shortwave, output_dir_name):
    """Creates figure showing overall model evaluation.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param for_shortwave: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if for_shortwave:
        input_file_suffixes_listlist = [
            SHORTWAVE_FLUX_INPUT_FILE_SUFFIXES, SHORTWAVE_HR_INPUT_FILE_SUFFIXES
        ]
    else:
        input_file_suffixes_listlist = [
            LONGWAVE_FLUX_INPUT_FILE_SUFFIXES, LONGWAVE_HR_INPUT_FILE_SUFFIXES
        ]

    concat_figure_file_names = [
        '{0:s}/overall_uq_evaluation_fluxes.jpg'.format(output_dir_name),
        '{0:s}/overall_uq_evaluation_heating_rates.jpg'.format(output_dir_name)
    ]

    for k in range(len(concat_figure_file_names)):
        panel_file_names = [
            '{0:s}/{1:s}'.format(input_dir_name, p)
            for p in input_file_suffixes_listlist[k]
        ]
        resized_panel_file_names = [
            '{0:s}/{1:s}'.format(output_dir_name, p.split('/')[-1])
            for p in input_file_suffixes_listlist[k]
        ]

        # Handles case where all panels are available except CRPS and CRPSS.
        if not any([os.path.isfile(f) for f in panel_file_names[-2:]]):
            panel_file_names = panel_file_names[:-2]
            resized_panel_file_names = resized_panel_file_names[:-2]

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

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_names[k]
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=resized_panel_file_names,
            output_file_name=concat_figure_file_names[k],
            num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_names[k],
            output_file_name=concat_figure_file_names[k],
            output_size_pixels=CONCAT_FIGURE_SIZE_PX
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        for_shortwave=bool(getattr(INPUT_ARG_OBJECT, FOR_SHORTWAVE_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
