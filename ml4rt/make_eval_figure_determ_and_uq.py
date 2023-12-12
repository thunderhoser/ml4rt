"""Creates 10-panel figure showing evaluation metrics for one model.

This includes evaluation metrics for both deterministic predictions and
uncertainty quantification (UQ).
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils
import example_utils

METRES_TO_MICRONS = 1e6

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 4
NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
MAKE_12PANEL_FIG_ARG_NAME = 'make_12panel_figure'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
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
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one figure for each.'
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
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_input_files_1wavelength(input_dir_name, make_12panel_figure,
                                 wavelength_metres):
    """Returns paths to input files for one wavelength.

    N = number of input files

    :param input_dir_name: Name of input directory.
    :param make_12panel_figure: See documentation at top of file.
    :param wavelength_metres: Wavelength.  Will assume that figures evaluating
        radiative transfer at this wavelength are to be concatenated.
    :return: input_file_names: length-N list of paths to input files.
    :return: panel_letters: length-N list of panel letters (each a 1-character
        string).
    """

    wavelength_microns = METRES_TO_MICRONS * wavelength_metres

    if make_12panel_figure:
        pathless_wavelengthless_file_names = [
            'evaluation/net-shortwave-flux-w-m02_attributes_new-model.jpg',
            'evaluation/shortwave-heating-rate-k-day01_large_error_frequency.jpg',
            'evaluation/shortwave-heating-rate-k-day01_catastrophic_error_frequency.jpg',
            'evaluation/shortwave-heating-rate-k-day01_attributes_new-model.jpg',
            'spread_vs_skill/spread_vs_skill_net-shortwave-flux-w-m02.jpg',
            'spread_vs_skill/spread_vs_skill_shortwave-heating-rate-k-day01.jpg',
            'evaluation/shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
            'discard_test_for_heating_rates/discard_test_net-shortwave-flux-w-m02.jpg',
            'discard_test_for_heating_rates/discard_test_shortwave-heating-rate-k-day01.jpg',
            'evaluation/shortwave-heating-rate-k-day01_bias_profile.jpg',
            'pit_histograms/pit_histogram_net-shortwave-flux-w-m02.jpg',
            'pit_histograms/pit_histogram_shortwave-heating-rate-k-day01.jpg'
        ]

        panel_letters = [
            'a', 'e', 'i', 'b', 'f', 'j', 'c', 'g', 'k', 'd', 'h', 'l'
        ]
    else:
        pathless_wavelengthless_file_names = [
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

        panel_letters = ['a', 'e', 'h', 'b', 'f', 'i', 'c', 'g', 'j', 'd']

    pathless_file_names = [
        '_'.join(
            [f.split('_')[0], '{0:.2f}microns'.format(wavelength_microns)] +
            f.split('_')[1:]
        )
        for f in pathless_wavelengthless_file_names
    ]

    input_file_names = [
        '{0:s}/{1:s}'.format(input_dir_name, f) for f in pathless_file_names
    ]

    return input_file_names, panel_letters


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


def _run(input_dir_name, make_12panel_figure, wavelengths_metres,
         output_dir_name):
    """Creates 10-panel figure showing evaluation metrics for one model.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param make_12panel_figure: Same.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for this_wavelength_metres in wavelengths_metres:
        panel_file_names, panel_letters = _get_input_files_1wavelength(
            input_dir_name=input_dir_name,
            make_12panel_figure=make_12panel_figure,
            wavelength_metres=this_wavelength_metres
        )
        resized_panel_file_names, _ = _get_input_files_1wavelength(
            input_dir_name=output_dir_name,
            make_12panel_figure=make_12panel_figure,
            wavelength_metres=this_wavelength_metres
        )
        resized_panel_file_names = [
            '{0:s}/{1:s}'.format(f.split('/')[0], f.split('/')[-1])
            for f in resized_panel_file_names
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
                x_offset_from_left_px=0,
                y_offset_from_top_px=TITLE_FONT_SIZE,
                text_string='({0:s})'.format(panel_letters[i])
            )
            imagemagick_utils.resize_image(
                input_file_name=resized_panel_file_names[i],
                output_file_name=resized_panel_file_names[i],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = (
            '{0:s}/overall_eval_determ_and_uq_{1:.2f}microns.jpg'
        ).format(
            output_dir_name,
            METRES_TO_MICRONS * this_wavelength_metres
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=resized_panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=NUM_PANEL_ROWS,
            num_panel_columns=NUM_PANEL_COLUMNS
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
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
