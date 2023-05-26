"""Splits predictions by perturbation type.

This script accounts for perturbations to 5 predictor variables:

- liquid-water content (LWC)
- ice-water content (IWC)
- specific humidity
- temperature
- ozone mixing ratio
"""

import os
import copy
import argparse
import warnings
import numpy
from ml4rt.utils import example_utils
from ml4rt.io import prediction_io
from ml4rt.utils import misc as misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RELEVANT_FIELD_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME,
    example_utils.TEMPERATURE_NAME,
    example_utils.O3_MIXING_RATIO_NAME
]

PREDICTION_FILE_ARG_NAME = 'input_perturbed_prediction_file_name'
CLEAN_EXAMPLE_DIR_ARG_NAME = 'input_clean_example_dir_name'
PERTURBED_EXAMPLE_DIR_ARG_NAME = 'input_perturbed_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all perturbed examples.  '
    'Will be read by `prediction_io.read_file`.'
)
CLEAN_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with model-input data (predictors and targets) for '
    'clean examples.  Files therein will be found by `example_io.find_file` '
    'and read by `example_io.read_file`.'
)
PERTURBED_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with model-input data (predictors and targets) for '
    'perturbed examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files for different perturbation types will be '
    'written here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLEAN_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=CLEAN_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PERTURBED_EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=PERTURBED_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(perturbed_prediction_file_name, clean_example_dir_name,
         perturbed_example_dir_name, output_dir_name):
    """Splits predictions by perturbation type.

    This is effectively the main method.

    :param perturbed_prediction_file_name: See documentation at top of file.
    :param clean_example_dir_name: Same.
    :param perturbed_example_dir_name: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...\n'.format(
        perturbed_prediction_file_name
    ))
    perturbed_prediction_dict = prediction_io.read_file(
        perturbed_prediction_file_name
    )

    perturbed_example_dict = misc_utils.get_raw_examples(
        example_file_name='', num_examples=int(1e12),
        example_dir_name=perturbed_example_dir_name,
        example_id_file_name=perturbed_prediction_file_name,
        ignore_sfc_temp_in_example_id=False,
        allow_missing_examples=False
    )[0]
    perturbed_example_dict = example_utils.subset_by_field(
        example_dict=perturbed_example_dict, field_names=RELEVANT_FIELD_NAMES
    )
    print(SEPARATOR_STRING)

    clean_example_dict, found_clean_example_flags = misc_utils.get_raw_examples(
        example_file_name='', num_examples=int(1e12),
        example_dir_name=clean_example_dir_name,
        example_id_file_name=perturbed_prediction_file_name,
        ignore_sfc_temp_in_example_id=True,
        allow_missing_examples=True
    )
    clean_example_dict = example_utils.subset_by_field(
        example_dict=clean_example_dict, field_names=RELEVANT_FIELD_NAMES
    )
    print(SEPARATOR_STRING)

    found_clean_example_indices = numpy.where(found_clean_example_flags)[0]

    if not numpy.all(found_clean_example_flags):
        warning_string = (
            'POTENTIAL ERROR: {0:d} of {1:d} desired clean examples cannot be '
            'found.'
        ).format(
            numpy.sum(found_clean_example_flags == False),
            len(found_clean_example_flags)
        )

        warnings.warn(warning_string)

    perturbed_example_dict = example_utils.subset_by_index(
        example_dict=perturbed_example_dict,
        desired_indices=found_clean_example_indices
    )
    clean_example_dict = example_utils.subset_by_index(
        example_dict=clean_example_dict,
        desired_indices=found_clean_example_indices
    )
    perturbed_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=perturbed_prediction_dict,
        desired_indices=found_clean_example_indices
    )

    num_examples = len(clean_example_dict[example_utils.EXAMPLE_IDS_KEY])

    for this_field_name in RELEVANT_FIELD_NAMES:
        perturbed_data_matrix = example_utils.get_field_from_dict(
            example_dict=perturbed_example_dict, field_name=this_field_name
        )
        clean_data_matrix = example_utils.get_field_from_dict(
            example_dict=clean_example_dict, field_name=this_field_name
        )
        unperturbed_flags = numpy.array([
            numpy.allclose(
                perturbed_data_matrix[i, :], clean_data_matrix[i, :],
                rtol=1e-3, equal_nan=True
            ) for i in range(num_examples)
        ], dtype=bool)

        perturbed_indices = numpy.where(numpy.invert(unperturbed_flags))[0]
        print('Number of examples with perturbed {0:s} = {1:d} of {2:d}'.format(
            this_field_name, len(perturbed_indices), num_examples
        ))

        this_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(perturbed_prediction_dict),
            desired_indices=perturbed_indices
        )

        this_output_file_name = '{0:s}/{1:s}/{2:s}'.format(
            output_dir_name, this_field_name,
            os.path.split(perturbed_prediction_file_name)[1]
        )
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            isotonic_model_file_name=
            this_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
            uncertainty_calib_model_file_name=this_prediction_dict[
                prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
            ],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        perturbed_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        clean_example_dir_name=getattr(
            INPUT_ARG_OBJECT, CLEAN_EXAMPLE_DIR_ARG_NAME
        ),
        perturbed_example_dir_name=getattr(
            INPUT_ARG_OBJECT, PERTURBED_EXAMPLE_DIR_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
