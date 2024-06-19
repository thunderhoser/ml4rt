"""Same as apply_neural_net.py but for one of Peter Ukkonen's neural nets."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NUM_EXAMPLES_PER_BATCH = 500

DUMMY_NORMALIZATION_FILE_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/'
    'examples_with_correct_vertical_coords/shortwave/training/'
    'learning_examples_for_norm_20180901-20191221.nc'
)

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The neural net will be applied only to'
    ' examples from `{0:s}` to `{1:s}`.'
).format(
    FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `prediction_io.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, first_time_string, last_time_string,
         output_file_name):
    """Same as apply_neural_net.py but for one of Peter Ukkonen's neural nets.

    This is effectively the main method.

    :param model_file_name: See documentation at top of this script.
    :param example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param output_file_name: Same.
    """

    # Process input args.
    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )

    # Read model and metadata.
    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    # Prepare input args for `neural_net.create_data_for_peter`.
    generator_option_dict = copy.deepcopy(
        metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name

    # Do the things.
    predictor_dict, vector_target_matrix, example_id_strings = (
        neural_net.create_data_for_peter(generator_option_dict)
    )
    vector_target_matrix = numpy.expand_dims(vector_target_matrix, axis=-2)
    print(SEPARATOR_STRING)

    example_id_strings, unique_indices = numpy.unique(
        numpy.array(example_id_strings), return_index=True
    )
    example_id_strings = example_id_strings.tolist()
    vector_target_matrix = vector_target_matrix[unique_indices, ...]
    for this_key in predictor_dict:
        predictor_dict[this_key] = predictor_dict[this_key][unique_indices, ...]

    these_keys = [
        'scalar_predictor_matrix',
        'vector_predictor_matrix',
        'toa_flux_input_matrix'
    ]
    predictor_matrices = [
        predictor_dict[k] for k in these_keys if k in predictor_dict
    ]
    if len(predictor_matrices) == 1:
        predictor_matrix_or_list = predictor_matrices[0]
    else:
        predictor_matrix_or_list = predictor_matrices

    del predictor_dict

    vector_prediction_matrix = neural_net.apply_model_for_peter(
        model_object=model_object,
        predictor_matrix_or_list=predictor_matrix_or_list,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        verbose=True,
        remove_fluxes=False
    )

    # Separate heating rates and fluxes.
    sfc_down_flux_prediction_matrix = vector_prediction_matrix[:, 0, :, 0, :]
    toa_up_flux_prediction_matrix = vector_prediction_matrix[:, -1, :, 1, :]
    scalar_prediction_matrix = numpy.stack(
        [sfc_down_flux_prediction_matrix, toa_up_flux_prediction_matrix],
        axis=-2
    )
    vector_prediction_matrix = vector_prediction_matrix[..., -1:, :]

    sfc_down_flux_target_matrix = vector_target_matrix[:, 0, :, 0]
    toa_up_flux_target_matrix = vector_target_matrix[:, -1, :, 1]
    scalar_target_matrix = numpy.stack(
        [sfc_down_flux_target_matrix, toa_up_flux_target_matrix],
        axis=-1
    )
    vector_target_matrix = vector_target_matrix[..., -1:]

    # Write output file.
    dummy_model_file_name = '{0:s}/model.keras'.format(
        os.path.split(output_file_name)[0]
    )

    print('Writing target (actual) and predicted values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=scalar_target_matrix,
        vector_target_matrix=vector_target_matrix,
        scalar_prediction_matrix=scalar_prediction_matrix,
        vector_prediction_matrix=vector_prediction_matrix,
        heights_m_agl=neural_net.HEIGHTS_FOR_PETER_M_AGL,
        target_wavelengths_metres=
        neural_net.TARGET_WAVELENGTHS_FOR_PETER_METRES,
        example_id_strings=example_id_strings,
        model_file_name=dummy_model_file_name,
        isotonic_model_file_name=None,
        uncertainty_calib_model_file_name=None,
        normalization_file_name=DUMMY_NORMALIZATION_FILE_NAME
    )

    new_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(dummy_model_file_name)[0],
        raise_error_if_missing=False
    )

    # TODO(thunderhoser): Will eventually need to generalize this for longwave
    # models.
    dummy_generator_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: 'foo',
        neural_net.BATCH_SIZE_KEY: 724,
        neural_net.SCALAR_PREDICTOR_NAMES_KEY: [],
        neural_net.VECTOR_PREDICTOR_NAMES_KEY: [],
        neural_net.SCALAR_TARGET_NAMES_KEY: [
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        ],
        neural_net.VECTOR_TARGET_NAMES_KEY:
            [example_utils.SHORTWAVE_HEATING_RATE_NAME],
        neural_net.HEIGHTS_KEY: neural_net.HEIGHTS_FOR_PETER_M_AGL,
        neural_net.TARGET_WAVELENGTHS_KEY:
            neural_net.TARGET_WAVELENGTHS_FOR_PETER_METRES,
        neural_net.FIRST_TIME_KEY: 0,
        neural_net.LAST_TIME_KEY: int(1e9),
        neural_net.NORMALIZATION_FILE_KEY: None,
        neural_net.NORMALIZE_PREDICTORS_KEY: False,
        neural_net.NORMALIZE_SCALAR_TARGETS_KEY: False,
        neural_net.NORMALIZE_VECTOR_TARGETS_KEY: False,
        neural_net.JOINED_OUTPUT_LAYER_KEY: False,
        neural_net.NUM_DEEP_SUPER_LAYERS_KEY: 0,
        neural_net.NORMALIZATION_FILE_FOR_MASK_KEY: None,
        neural_net.MIN_HEATING_RATE_FOR_MASK_KEY: None,
        neural_net.MIN_FLUX_FOR_MASK_KEY: None
    }

    md = metadata_dict
    md[neural_net.TRAINING_OPTIONS_KEY] = dummy_generator_option_dict
    md[neural_net.VALIDATION_OPTIONS_KEY] = dummy_generator_option_dict

    print('Writing new metafile to: "{0:s}"...'.format(new_metafile_name))
    neural_net._write_metafile(
        dill_file_name=new_metafile_name,
        num_epochs=md[neural_net.NUM_EPOCHS_KEY],
        num_training_batches_per_epoch=md[neural_net.NUM_TRAINING_BATCHES_KEY],
        training_option_dict=md[neural_net.TRAINING_OPTIONS_KEY],
        num_validation_batches_per_epoch=
        md[neural_net.NUM_VALIDATION_BATCHES_KEY],
        validation_option_dict=md[neural_net.VALIDATION_OPTIONS_KEY],
        loss_function_or_dict=md[neural_net.LOSS_FUNCTION_OR_DICT_KEY],
        plateau_lr_multiplier=md[neural_net.PLATEAU_LR_MUTIPLIER_KEY],
        early_stopping_patience_epochs=
        md[neural_net.EARLY_STOPPING_PATIENCE_KEY],
        dense_architecture_dict=None,
        cnn_architecture_dict=None,
        bnn_architecture_dict=None,
        u_net_architecture_dict=None,
        u_net_plusplus_architecture_dict=None,
        u_net_3plus_architecture_dict=None,
        use_ryan_architecture=True
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
