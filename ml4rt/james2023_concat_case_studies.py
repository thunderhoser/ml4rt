"""Concatenates case-study figures for JAMES 2023 paper.

USE ONCE (well, maybe a few times) AND DESTROY.
"""

import os
import sys
import glob
import argparse
from PIL import Image
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import imagemagick_utils
import example_utils

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 100
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

MODEL_DIR_SUFFIXES_CLEAN_TRAINING = [
    'shortwave_mme_no_crps_experiment01/multimodel_ensemble_50x1',
    'shortwave_mme_experiment01/multimodel_ensemble_50',

    'shortwave_bnn_experiment00/'
    'num-first-layer-channels=064_num-bayesian-skip-layers=2_'
    'num-bayesian-upconv-layers=2_num-bayesian-dense-layers=3_'
    'bayesian-layer-type=reparameterization/model',

    'shortwave_bnn_crps_experiment00/'
    'num-first-layer-channels=064_num-bayesian-skip-layers=2_'
    'num-bayesian-upconv-layers=1_num-bayesian-dense-layers=3_'
    'bayesian-layer-type=reparameterization/model',

    'shortwave_crps_experiment00/model'
]

MODEL_DIR_SUFFIXES_PERTURBED_TRAINING = [
    'shortwave_mme_no_crps_experiment02/multimodel_ensemble_50x1',

    'shortwave_mme_experiment02/multimodel_ensemble_50',

    'shortwave_bnn_experiment01/'
    'num-first-layer-channels=128_num-bayesian-skip-layers=1_'
    'num-bayesian-upconv-layers=2_num-bayesian-dense-layers=2_'
    'bayesian-layer-type=reparameterization/model'

    'shortwave_bnn_crps_experiment01/'
    'num-first-layer-channels=064_num-bayesian-skip-layers=2_'
    'num-bayesian-upconv-layers=2_num-bayesian-dense-layers=1_'
    'bayesian-layer-type=flipout/model',

    'shortwave_crps_experiment01/model'
]

MODEL_DESCRIPTION_STRINGS = [
    'mme_only', 'mme_crps', 'bnn_only', 'bnn_crps', 'crps_only'
]

ALL_EXPERIMENTS_DIR_ARG_NAME = 'input_all_experiments_dir_name'
USE_CLEAN_TRAINING_ARG_NAME = 'use_clean_training'
USE_VALIDATION_CASES_ARG_NAME = 'use_validation_cases'
USE_HEIGHT_AVG_EXTREMES_ARG_NAME = 'use_height_avg_extremes'
EXAMPLE_IDS_ARG_NAME = 'example_id_strings'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

ALL_EXPERIMENTS_DIR_HELP_STRING = (
    'Name of directory with models for all experiments.'
)
USE_CLEAN_TRAINING_HELP_STRING = (
    'Boolean flag.  If True (False), will use models trained with clean '
    '(perturbed) data.'
)
USE_VALIDATION_CASES_HELP_STRING = (
    'Boolean flag.  If True (False), will use case studies in the validation '
    '(testing) data.'
)
USE_HEIGHT_AVG_EXTREMES_HELP_STRING = (
    'Boolean flag.  If True (False), will use height-averaged (single-height) '
    'extremes.'
)
EXAMPLE_IDS_HELP_STRING = (
    'List of example IDs.  Will create one figure for each example.  Example '
    'of an example ID: '
    '"lat=02.284413-long=-156.796875-zenith-angle-rad=1.248265-time=1579111200-'
    'atmo=3-albedo=0.157677-temp-10m-kelvins=300.000610".  '
    'If you would rather work with random examples, leave this argument alone '
    'and use {0:s} instead.'
).format(NUM_EXAMPLES_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to select randomly.  Will create one figure for each '
    'example.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ALL_EXPERIMENTS_DIR_ARG_NAME, type=str, required=True,
    help=ALL_EXPERIMENTS_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_CLEAN_TRAINING_ARG_NAME, type=int, required=True,
    help=USE_CLEAN_TRAINING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_VALIDATION_CASES_ARG_NAME, type=int, required=True,
    help=USE_VALIDATION_CASES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_HEIGHT_AVG_EXTREMES_ARG_NAME, type=int, required=True,
    help=USE_HEIGHT_AVG_EXTREMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_IDS_ARG_NAME, type=str, nargs='+', required=False,
    default=[''], help=EXAMPLE_IDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
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


def _concat_panels_one_example(
        all_experiments_dir_name, use_clean_training, use_validation_cases,
        use_height_avg_extremes, example_id_string, output_dir_name):
    """Creates figure for one example.

    :param all_experiments_dir_name: See documentation at top of file.
    :param use_clean_training: Same.
    :param use_validation_cases: Same.
    :param use_height_avg_extremes: Same.
    :param example_id_string: Example ID.
    :param output_dir_name: See documentation at top of file.
    """

    model_dir_suffixes = (
        MODEL_DIR_SUFFIXES_CLEAN_TRAINING if use_clean_training
        else MODEL_DIR_SUFFIXES_PERTURBED_TRAINING
    )
    val_or_testing_string = (
        'validation_perturbed_for_uq' if use_validation_cases
        else 'testing_perturbed_for_uq'
    )
    height_processing_string = (
        'extreme_hr_avg-by-height' if use_height_avg_extremes
        else 'extreme_hr_1height'
    )

    panel_file_names = [
        (
            '{0:s}/{1:s}/{2:s}/{3:s}/predictions_large-heating-rate/'
            '{4:s}_shortwave-heating-rate-k-day01.jpg'
        ).format(
            all_experiments_dir_name,
            d,
            val_or_testing_string,
            height_processing_string,
            example_id_string
        )
        for d in model_dir_suffixes
    ]

    lettered_panel_file_names = [
        '{0:s}/{1:s}/{2:s}_shortwave-heating-rate-k-day01.jpg'.format(
            output_dir_name, s, example_id_string
        )
        for s in MODEL_DESCRIPTION_STRINGS
    ]

    letter_label = 'b'

    for i in range(len(panel_file_names)):
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=lettered_panel_file_names[i],
            border_width_pixels=0
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _overlay_text(
            image_file_name=lettered_panel_file_names[i],
            x_offset_from_left_px=TITLE_FONT_SIZE,
            y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=lettered_panel_file_names[i],
            output_file_name=lettered_panel_file_names[i],
            border_width_pixels=0
        )
        imagemagick_utils.resize_image(
            input_file_name=lettered_panel_file_names[i],
            output_file_name=lettered_panel_file_names[i],
            output_size_pixels=int(PANEL_SIZE_PX)
        )

    concat_prediction_figure_file_name = (
        '{0:s}/prediction_comparisons/{1:s}.jpg'
    ).format(output_dir_name, example_id_string)

    print('Concatenating panels to: "{0:s}"...'.format(
        concat_prediction_figure_file_name
    ))

    imagemagick_utils.concatenate_images(
        input_file_names=lettered_panel_file_names,
        output_file_name=concat_prediction_figure_file_name,
        num_panel_rows=3, num_panel_columns=2, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_prediction_figure_file_name,
        output_file_name=concat_prediction_figure_file_name,
        border_width_pixels=0
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_prediction_figure_file_name,
        output_file_name=concat_prediction_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    predictor_file_name = (
        '{0:s}/{1:s}/{2:s}/{3:s}/predictions_large-heating-rate/full_examples/'
        '{4:s}.jpg'
    ).format(
        all_experiments_dir_name,
        model_dir_suffixes[0],
        val_or_testing_string,
        height_processing_string,
        example_id_string
    )

    lettered_predictor_file_name = '{0:s}/predictors/{1:s}.jpg'.format(
        output_dir_name, example_id_string
    )

    imagemagick_utils.trim_whitespace(
        input_file_name=predictor_file_name,
        output_file_name=lettered_predictor_file_name,
        border_width_pixels=0
    )
    _overlay_text(
        image_file_name=lettered_predictor_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE,
        y_offset_from_top_px=TITLE_FONT_SIZE,
        text_string='(a)'
    )

    image_width_px = Image.open(lettered_predictor_file_name).size[0]
    image_half_width_px = int(numpy.round(0.5 * image_width_px))

    _overlay_text(
        image_file_name=lettered_predictor_file_name,
        x_offset_from_left_px=TITLE_FONT_SIZE + image_half_width_px,
        y_offset_from_top_px=TITLE_FONT_SIZE,
        text_string='(b)'
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=lettered_predictor_file_name,
        output_file_name=lettered_predictor_file_name,
        border_width_pixels=0
    )

    predictor_image_matrix = Image.open(lettered_predictor_file_name)
    predictor_image_width_px, predictor_image_height_px = (
        predictor_image_matrix.size
    )
    predictor_image_size_px = (
        predictor_image_width_px * predictor_image_height_px
    )

    prediction_image_width_px = Image.open(
        concat_prediction_figure_file_name
    ).size[0]

    width_ratio = float(prediction_image_width_px) / predictor_image_width_px
    predictor_image_size_px = int(numpy.round(
        predictor_image_size_px * (width_ratio ** 2)
    ))

    imagemagick_utils.resize_image(
        input_file_name=lettered_predictor_file_name,
        output_file_name=lettered_predictor_file_name,
        output_size_pixels=predictor_image_size_px
    )

    lettered_panel_file_names = [
        lettered_predictor_file_name, concat_prediction_figure_file_name
    ]
    concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=lettered_panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=1, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=0
    )


def _run(all_experiments_dir_name, use_clean_training, use_validation_cases,
         use_height_avg_extremes, example_id_strings, num_examples,
         output_dir_name):
    """Concatenates case-study figures for JAMES 2023 paper.

    This is effectively the main method.

    :param all_experiments_dir_name: See documentation at top of file.
    :param use_clean_training: Same.
    :param use_validation_cases: Same.
    :param use_height_avg_extremes: Same.
    :param example_id_strings: Same.
    :param num_examples: Same.
    :param output_dir_name: Same.
    """

    if len(example_id_strings) == 1 and example_id_strings[0] == '':
        example_id_strings = None

    if example_id_strings is None:
        error_checking.assert_is_greater(num_examples, 0)

        first_model_dir_name = '{0:s}/{1:s}'.format(
            all_experiments_dir_name,
            MODEL_DIR_SUFFIXES_CLEAN_TRAINING[0] if use_clean_training
            else MODEL_DIR_SUFFIXES_PERTURBED_TRAINING[0]
        )
        val_or_testing_string = (
            'validation_perturbed_for_uq' if use_validation_cases
            else 'testing_perturbed_for_uq'
        )
        height_processing_string = (
            'extreme_hr_avg-by-height' if use_height_avg_extremes
            else 'extreme_hr_1height'
        )

        figure_file_pattern = (
            '{0:s}/{1:s}/{2:s}/predictions_large-heating-rate/'
            '*_shortwave-heating-rate-k-day01.jpg'
        ).format(
            first_model_dir_name,
            val_or_testing_string,
            height_processing_string
        )
        print(figure_file_pattern)

        figure_file_names = glob.glob(figure_file_pattern)
        num_figures = len(figure_file_names)
        example_id_strings = []

        for i in range(num_figures):
            this_pathless_file_name = os.path.split(figure_file_names[i])[1]
            this_example_id_string = '_'.join(
                this_pathless_file_name.split('_')[:-1]
            )

            try:
                example_utils.parse_example_ids([this_example_id_string])
                example_id_strings.append(this_example_id_string)
            except:
                pass

        num_examples_found = len(example_id_strings)

        if num_examples > num_examples_found:
            good_indices = numpy.linspace(
                0, num_examples_found - 1, num=num_examples_found, dtype=int
            )
            good_indices = numpy.random.choice(
                good_indices, size=num_examples, replace=False
            )
            example_id_strings = [example_id_strings[k] for k in good_indices]

    for this_example_id_string in example_id_strings:
        _concat_panels_one_example(
            all_experiments_dir_name=all_experiments_dir_name,
            use_clean_training=use_clean_training,
            use_validation_cases=use_validation_cases,
            use_height_avg_extremes=use_height_avg_extremes,
            example_id_string=this_example_id_string,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        all_experiments_dir_name=getattr(
            INPUT_ARG_OBJECT, ALL_EXPERIMENTS_DIR_ARG_NAME
        ),
        use_clean_training=bool(
            getattr(INPUT_ARG_OBJECT, USE_CLEAN_TRAINING_ARG_NAME)
        ),
        use_validation_cases=bool(
            getattr(INPUT_ARG_OBJECT, USE_VALIDATION_CASES_ARG_NAME)
        ),
        use_height_avg_extremes=bool(
            getattr(INPUT_ARG_OBJECT, USE_HEIGHT_AVG_EXTREMES_ARG_NAME)
        ),
        example_id_strings=getattr(INPUT_ARG_OBJECT, EXAMPLE_IDS_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
