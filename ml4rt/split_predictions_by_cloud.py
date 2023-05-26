"""Splits predictions by cloud regime.

This script allows for four different cloud regimes:

- no cloud (clear sky)
- single-layer cloud
- multi-layer cloud
- fog (cloud reaching surface)
"""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_utils
import prediction_io
import misc as misc_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

KG_TO_GRAMS = 1e3

MIN_LAYERS_BY_FILE = numpy.array([0, 1, 2], dtype=int)
MAX_LAYERS_BY_FILE = numpy.array([0, 1, 1e12], dtype=int)
SUBDIR_NAMES = ['no_cloud', 'single_layer_cloud', 'multi_layer_cloud']

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FOR_ICE_ARG_NAME = 'for_ice'
MIN_PATH_ARG_NAME = 'min_path_kg_m02'
INCLUDE_FOG_ARG_NAME = 'include_fog'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all cloud regimes.  Will be'
    ' read by `prediction_io.read_file`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
FOR_ICE_HELP_STRING = (
    'Boolean flag.  If 1 (0), cloud regimes will be based on ice (liquid) '
    'cloud.'
)
MIN_PATH_HELP_STRING = 'Minimum water path (kg m^-2) for each cloud layer.'
INCLUDE_FOG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will (not) include fog in cloud regimes.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Files for different cloud regimes will be '
    'written here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FOR_ICE_ARG_NAME, type=int, required=False, default=0,
    help=FOR_ICE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PATH_ARG_NAME, type=float, required=False, default=0.05,
    help=MIN_PATH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_FOG_ARG_NAME, type=int, required=False, default=0,
    help=INCLUDE_FOG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, example_dir_name, for_ice, min_path_kg_m02,
         include_fog, output_dir_name):
    """Splits predictions by cloud regime.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param for_ice: Same.
    :param min_path_kg_m02: Same.
    :param include_fog: Same.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...\n'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    example_dict = misc_utils.get_raw_examples(
        example_file_name='', num_examples=int(1e12),
        example_dir_name=example_dir_name, example_id_file_name=input_file_name
    )[0]
    print(SEPARATOR_STRING)

    cloud_layer_counts = example_utils.find_cloud_layers(
        example_dict=example_dict, min_path_kg_m02=min_path_kg_m02,
        for_ice=for_ice, fog_only=False
    )[-1]

    unique_cloud_layer_counts, unique_example_counts = numpy.unique(
        cloud_layer_counts, return_counts=True
    )

    for i in range(len(unique_cloud_layer_counts)):
        print((
            'Number of examples with {0:d} cloud layers '
            '({1:s}-water path >= {2:.1f} g m^-2) = {3:d}'
        ).format(
            unique_cloud_layer_counts[i],
            'ice' if for_ice else 'liquid',
            KG_TO_GRAMS * min_path_kg_m02,
            unique_example_counts[i]
        ))

    print(SEPARATOR_STRING)

    num_output_files = len(MIN_LAYERS_BY_FILE)

    for k in range(num_output_files):
        these_indices = numpy.where(numpy.logical_and(
            cloud_layer_counts >= MIN_LAYERS_BY_FILE[k],
            cloud_layer_counts <= MAX_LAYERS_BY_FILE[k]
        ))[0]

        this_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=these_indices
        )

        this_output_file_name = '{0:s}/{1:s}/{2:s}'.format(
            output_dir_name, SUBDIR_NAMES[k], os.path.split(input_file_name)[1]
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

    if not include_fog:
        return

    print(SEPARATOR_STRING)
    cloud_layer_counts = example_utils.find_cloud_layers(
        example_dict=example_dict, min_path_kg_m02=min_path_kg_m02,
        for_ice=for_ice, fog_only=True
    )[-1]
    print(SEPARATOR_STRING)

    fog_indices = numpy.where(cloud_layer_counts > 0)[0]
    print('Number of examples with {0:s} fog = {1:d}'.format(
        'ice' if for_ice else 'liquid',
        len(fog_indices)
    ))

    this_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=fog_indices
    )

    this_output_file_name = '{0:s}/fog/predictions.nc'.format(output_dir_name)
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
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        for_ice=bool(getattr(INPUT_ARG_OBJECT, FOR_ICE_ARG_NAME)),
        min_path_kg_m02=getattr(INPUT_ARG_OBJECT, MIN_PATH_ARG_NAME),
        include_fog=bool(getattr(INPUT_ARG_OBJECT, INCLUDE_FOG_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
