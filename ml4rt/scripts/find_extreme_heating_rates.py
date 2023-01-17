"""Finds extreme heating rates (largest, smallest, best/worst predicted)."""

import os
import copy
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.utils import example_utils
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
FOR_SHORTWAVE_ARG_NAME = 'for_shortwave'
AVERAGE_OVER_HEIGHT_ARG_NAME = 'average_over_height'
SCALE_BY_CLIMO_ARG_NAME = 'scale_by_climo'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_set'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted values.  Will be read '
    'by `prediction_io.write_file`.'
)
FOR_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will find extreme shortwave (longwave) heating '
    'rates.'
)
AVERAGE_OVER_HEIGHT_HELP_STRING = (
    'Boolean flag.  If 1, will average errors over height for each profile.  '
    'If 0, will look for height with worst error in each profile.'
)
SCALE_BY_CLIMO_HELP_STRING = (
    'Boolean flag.  If 1, will scale error at each height z by climatology '
    '(average heating rate at height z in training data).'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples in each set (of either best or worst predictions).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Each set (of either best or worst predictions) '
    'will be written here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FOR_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=FOR_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + AVERAGE_OVER_HEIGHT_ARG_NAME, type=int, required=True,
    help=AVERAGE_OVER_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCALE_BY_CLIMO_ARG_NAME, type=int, required=True,
    help=SCALE_BY_CLIMO_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_prediction_file_name, for_shortwave, average_over_height,
         scale_by_climo, num_examples_per_set, output_dir_name):
    """Finds best and worst heating-rate predictions.

    This is effectively the main method.

    :param input_prediction_file_name: See documentation at top of file.
    :param for_shortwave: Same.
    :param average_over_height: Same.
    :param scale_by_climo: Same.
    :param num_examples_per_set: Same.
    :param output_dir_name: Same.
    """

    # TODO(thunderhoser): Maybe allow specific height again (e.g., 15 km).

    error_checking.assert_is_greater(num_examples_per_set, 0)
    scale_by_climo = scale_by_climo and not average_over_height

    print('Reading data from: "{0:s}"...'.format(input_prediction_file_name))
    prediction_dict = prediction_io.read_file(input_prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0]
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    hr_index = vector_target_names.index(
        example_utils.SHORTWAVE_HEATING_RATE_NAME if for_shortwave
        else example_utils.LONGWAVE_HEATING_RATE_NAME
    )

    target_matrix_k_day01 = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][..., hr_index]
    )
    prediction_matrix_k_day01 = numpy.mean(
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][..., hr_index, :],
        axis=-1
    )

    bias_matrix = prediction_matrix_k_day01 - target_matrix_k_day01
    absolute_error_matrix = numpy.absolute(bias_matrix)

    if average_over_height:
        bias_matrix = numpy.mean(bias_matrix, axis=1, keepdims=True)
        absolute_error_matrix = numpy.mean(
            absolute_error_matrix, axis=1, keepdims=True
        )

    if scale_by_climo:
        if prediction_dict[prediction_io.NORMALIZATION_FILE_KEY] is None:
            normalization_file_name = (
                generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
            )
        else:
            normalization_file_name = (
                prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
            )

        print((
            'Reading training examples (for climatology) from: "{0:s}"...'
        ).format(
            normalization_file_name
        ))

        training_example_dict = example_io.read_file(normalization_file_name)
        training_example_dict = example_utils.subset_by_field(
            example_dict=training_example_dict,
            field_names=[
                example_utils.SHORTWAVE_HEATING_RATE_NAME if for_shortwave
                else example_utils.LONGWAVE_HEATING_RATE_NAME
            ]
        )

        # Take absolute values, in case longwave.
        training_example_dict[example_utils.VECTOR_TARGET_VALS_KEY] = (
            numpy.absolute(
                training_example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
            )
        )

        if prediction_dict[prediction_io.NORMALIZATION_FILE_KEY] is None:
            heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]
        else:
            heights_m_agl = training_example_dict[example_utils.HEIGHTS_KEY]

        training_example_dict = example_utils.subset_by_height(
            example_dict=training_example_dict, heights_m_agl=heights_m_agl
        )

        dummy_example_dict = {
            example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
            example_utils.VECTOR_PREDICTOR_NAMES_KEY: [],
            example_utils.SCALAR_TARGET_NAMES_KEY: [],
            example_utils.VECTOR_TARGET_NAMES_KEY: [
                example_utils.SHORTWAVE_HEATING_RATE_NAME if for_shortwave
                else example_utils.LONGWAVE_HEATING_RATE_NAME
            ],
            example_utils.HEIGHTS_KEY: heights_m_agl
        }

        mean_training_example_dict = normalization.create_mean_example(
            new_example_dict=dummy_example_dict,
            training_example_dict=training_example_dict
        )
        climo_matrix_k_day01 = mean_training_example_dict[
            example_utils.VECTOR_TARGET_VALS_KEY
        ][..., 0]

        bias_matrix = bias_matrix / climo_matrix_k_day01
        absolute_error_matrix = absolute_error_matrix / climo_matrix_k_day01

    print(SEPARATOR_STRING)
    high_bias_indices, low_bias_indices, low_abs_error_indices = (
        misc_utils.find_best_and_worst_predictions(
            bias_matrix=bias_matrix,
            absolute_error_matrix=absolute_error_matrix,
            num_examples_per_set=num_examples_per_set
        )
    )
    print(SEPARATOR_STRING)

    high_bias_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=high_bias_indices
    )
    high_bias_file_name = (
        '{0:s}/predictions_high-bias.nc'.format(output_dir_name)
    )

    print('Writing examples with greatest positive bias to: "{0:s}"...'.format(
        high_bias_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=high_bias_file_name,
        scalar_target_matrix=
        high_bias_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        high_bias_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        high_bias_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        high_bias_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=high_bias_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        high_bias_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=high_bias_prediction_dict[prediction_io.MODEL_FILE_KEY],
        normalization_file_name=
        high_bias_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    low_bias_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_bias_indices
    )
    low_bias_file_name = (
        '{0:s}/predictions_low-bias.nc'.format(output_dir_name)
    )

    print('Writing examples with greatest negative bias to: "{0:s}"...'.format(
        low_bias_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_bias_file_name,
        scalar_target_matrix=
        low_bias_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        low_bias_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        low_bias_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        low_bias_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=low_bias_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        low_bias_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=low_bias_prediction_dict[prediction_io.MODEL_FILE_KEY],
        normalization_file_name=
        low_bias_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    low_abs_error_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=low_abs_error_indices
    )
    low_abs_error_file_name = (
        '{0:s}/predictions_low-absolute-error.nc'.format(output_dir_name)
    )

    print('Writing examples with smallest absolute error to: "{0:s}"...'.format(
        low_abs_error_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=low_abs_error_file_name,
        scalar_target_matrix=
        low_abs_error_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        low_abs_error_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        low_abs_error_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        low_abs_error_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=low_abs_error_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        low_abs_error_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=
        low_abs_error_prediction_dict[prediction_io.MODEL_FILE_KEY],
        normalization_file_name=
        low_abs_error_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    if scale_by_climo:
        return

    if average_over_height:
        mean_targets_k_day01 = numpy.mean(
            numpy.absolute(target_matrix_k_day01), axis=1
        )
        sort_indices = numpy.argsort(-1 * mean_targets_k_day01)
    else:
        max_targets_k_day01 = numpy.max(
            numpy.absolute(target_matrix_k_day01), axis=1
        )
        sort_indices = numpy.argsort(-1 * max_targets_k_day01)

    large_hr_indices = sort_indices[:num_examples_per_set]
    large_hr_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=large_hr_indices
    )
    large_hr_file_name = (
        '{0:s}/predictions_large-heating-rate.nc'.format(output_dir_name)
    )

    print('Writing examples with greatest heating rate to: "{0:s}"...'.format(
        large_hr_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=large_hr_file_name,
        scalar_target_matrix=
        large_hr_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        large_hr_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        large_hr_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        large_hr_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=large_hr_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        large_hr_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=large_hr_prediction_dict[prediction_io.MODEL_FILE_KEY],
        normalization_file_name=
        large_hr_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )

    if not average_over_height:
        return

    mean_targets_k_day01 = numpy.mean(
        numpy.absolute(target_matrix_k_day01), axis=1
    )
    sort_indices = numpy.argsort(mean_targets_k_day01)
    small_hr_indices = sort_indices[:num_examples_per_set]

    small_hr_prediction_dict = prediction_io.subset_by_index(
        prediction_dict=copy.deepcopy(prediction_dict),
        desired_indices=small_hr_indices
    )
    small_hr_file_name = (
        '{0:s}/predictions_small-heating-rate.nc'.format(output_dir_name)
    )

    print('Writing examples with smallest heating rate to: "{0:s}"...'.format(
        small_hr_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=small_hr_file_name,
        scalar_target_matrix=
        small_hr_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
        vector_target_matrix=
        small_hr_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        scalar_prediction_matrix=
        small_hr_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
        vector_prediction_matrix=
        small_hr_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
        heights_m_agl=small_hr_prediction_dict[prediction_io.HEIGHTS_KEY],
        example_id_strings=
        small_hr_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        model_file_name=small_hr_prediction_dict[prediction_io.MODEL_FILE_KEY],
        normalization_file_name=
        small_hr_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_prediction_file_name=getattr(
            INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME
        ),
        for_shortwave=bool(getattr(INPUT_ARG_OBJECT, FOR_SHORTWAVE_ARG_NAME)),
        average_over_height=bool(
            getattr(INPUT_ARG_OBJECT, AVERAGE_OVER_HEIGHT_ARG_NAME)
        ),
        scale_by_climo=bool(getattr(INPUT_ARG_OBJECT, SCALE_BY_CLIMO_ARG_NAME)),
        num_examples_per_set=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME),
    )
