"""Creates figure showing model evaluation by cloud regime."""

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
import plot_evaluation

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
FOR_SHORTWAVE_ARG_NAME = 'for_shortwave'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
INCLUDE_FOG_ARG_NAME = 'include_fog'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing evaluation figures created by '
    'plot_evaluation.py.  This script will panel some of those figures '
    'together.'
)
FOR_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will make figure with shortwave (longwave) '
    'errors.'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one figure for each.'
)
INCLUDE_FOG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) include fog as a cloud regime.'
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
    '--' + FOR_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=FOR_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_FOG_ARG_NAME, type=int, required=False, default=0,
    help=INCLUDE_FOG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_input_files_1wavelength(input_dir_name, for_shortwave,
                                 wavelength_metres, include_fog):
    """Returns paths to input files for one wavelength.

    :param input_dir_name: Name of input directory.
    :param for_shortwave: Boolean flag.  If True (False), will assume that
        figures evaluating shortwave (longwave) are to be concatenated.
    :param wavelength_metres: Wavelength.  Will assume that figures evaluating
        radiative transfer at this wavelength are to be concatenated.
    :param include_fog: See documentation at top of this script.
    :return: input_file_names: 1-D list of paths to input files.
    """

    pathless_wavelengthless_file_names = [
        'shortwave-surface-down-flux-w-m02_attributes_multi-layer-cloud.jpg',
        'shortwave-toa-up-flux-w-m02_attributes_multi-layer-cloud.jpg',
        'net-shortwave-flux-w-m02_attributes_multi-layer-cloud.jpg',
        'shortwave-heating-rate-k-day01_bias_profile.jpg',
        'shortwave-heating-rate-k-day01_mean-absolute-error_profile.jpg',
        'shortwave-heating-rate-k-day01_mae-skill-score_profile.jpg',
        'shortwave-heating-rate-k-day01_attributes_no-cloud.jpg',
        'shortwave-heating-rate-k-day01_attributes_single-layer-cloud.jpg',
        'shortwave-heating-rate-k-day01_attributes_multi-layer-cloud.jpg'
    ]

    if include_fog:
        pathless_wavelengthless_file_names.append(
            'shortwave-heating-rate-k-day01_attributes_fog.jpg'
        )

    if not for_shortwave:
        pathless_wavelengthless_file_names = [
            f.replace('shortwave', 'longwave')
            for f in pathless_wavelengthless_file_names
        ]

    pathless_file_names = [
        '_'.join(
            [f.split('_')[0], plot_evaluation.wavelength_to_string(wavelength_metres)] +
            f.split('_')[1:]
        )
        for f in pathless_wavelengthless_file_names
    ]

    return [
        '{0:s}/{1:s}'.format(input_dir_name, f) for f in pathless_file_names
    ]


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


def _run(input_dir_name, wavelengths_metres, for_shortwave, include_fog,
         output_dir_name):
    """Creates figure showing overall model evaluation.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param wavelengths_metres: Same.
    :param for_shortwave: Same.
    :param include_fog: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for this_wavelength_metres in wavelengths_metres:
        panel_file_names = _get_input_files_1wavelength(
            input_dir_name=input_dir_name,
            for_shortwave=for_shortwave,
            wavelength_metres=this_wavelength_metres,
            include_fog=include_fog
        )
        resized_panel_file_names = _get_input_files_1wavelength(
            input_dir_name=output_dir_name,
            for_shortwave=for_shortwave,
            wavelength_metres=this_wavelength_metres,
            include_fog=include_fog
        )

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
                x_offset_from_left_px=0,
                y_offset_from_top_px=TITLE_FONT_SIZE,
                text_string='({0:s})'.format(letter_label)
            )
            imagemagick_utils.resize_image(
                input_file_name=resized_panel_file_names[i],
                output_file_name=resized_panel_file_names[i],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = (
            '{0:s}/evaluation_by_cloud_regime_{1:s}microns.jpg'
        ).format(
            output_dir_name,
            plot_evaluation.wavelength_to_string(this_wavelength_metres)
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))

        num_panel_rows = int(numpy.ceil(
            float(len(panel_file_names)) / NUM_PANEL_COLUMNS
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=resized_panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows,
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
        for_shortwave=bool(getattr(INPUT_ARG_OBJECT, FOR_SHORTWAVE_ARG_NAME)),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        include_fog=bool(getattr(INPUT_ARG_OBJECT, INCLUDE_FOG_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
