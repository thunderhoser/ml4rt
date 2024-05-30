"""Contains list of input arguments for training a neural net."""

from ml4rt.utils import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
INPUT_MODEL_FILE_ARG_NAME = 'input_model_file_name'
OUTPUT_MODEL_DIR_ARG_NAME = 'output_model_dir_name'

USE_GENERATOR_FOR_TRAIN_ARG_NAME = 'use_generator_for_training'
USE_GENERATOR_FOR_VALIDN_ARG_NAME = 'use_generator_for_validn'
JOINED_OUTPUT_LAYER_ARG_NAME = 'joined_output_layer'
NUM_DEEP_SUPER_LAYERS_ARG_NAME = 'num_deep_supervision_layers'

PREDICTOR_NAMES_ARG_NAME = 'predictor_names'
TARGET_NAMES_ARG_NAME = 'target_names'
HEIGHTS_ARG_NAME = 'heights_m_agl'
TARGET_WAVELENGTHS_ARG_NAME = 'target_wavelengths_metres'
FIRST_TRAIN_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAIN_TIME_ARG_NAME = 'last_training_time_string'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'

NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
NORMALIZE_PREDICTORS_ARG_NAME = 'normalize_predictors'
NORMALIZE_SCALAR_TARGETS_ARG_NAME = 'normalize_scalar_targets'
NORMALIZE_VECTOR_TARGETS_ARG_NAME = 'normalize_vector_targets'

NORMALIZATION_FILE_FOR_MASK_ARG_NAME = 'normalization_file_name_for_mask'
MIN_HEATING_RATE_FOR_MASK_ARG_NAME = 'min_heating_rate_for_mask_k_day01'
MIN_FLUX_FOR_MASK_ARG_NAME = 'min_flux_for_mask_w_m02'

BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validn_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'
EARLY_STOPPING_PATIENCE_ARG_NAME = 'early_stopping_patience_epochs'

TRAINING_DIR_HELP_STRING = (
    'Name of directory with training examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring) examples.'
).format(
    TRAINING_DIR_ARG_NAME
)
INPUT_MODEL_FILE_HELP_STRING = (
    'Path to file with untrained model (defining architecture, optimizer, and '
    'loss function).  Will be read by `neural_net.read_model`.'
)
OUTPUT_MODEL_DIR_HELP_STRING = (
    'Name of output directory.  Model will be saved here.'
)

USE_GENERATOR_FOR_TRAIN_HELP_STRING = (
    'Boolean flag.  If 1, will use generator for training data.  If 0, will '
    'load all training data into memory at once.'
)
USE_GENERATOR_FOR_VALIDN_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring) data.'
).format(
    USE_GENERATOR_FOR_TRAIN_ARG_NAME
)
JOINED_OUTPUT_LAYER_HELP_STRING = (
    'Boolean flag.  If 1, model has one output layer for both heating rates and'
    ' fluxes.'
)
NUM_DEEP_SUPER_LAYERS_HELP_STRING = 'Number of deep-supervision layers.'

PREDICTOR_NAMES_HELP_STRING = (
    'List of predictor variables.  Each must be accepted by '
    '`example_utils.check_field_name`.'
)
TARGET_NAMES_HELP_STRING = (
    'List of target variables.  Each must be accepted by '
    '`example_utils.check_field_name`.'
)
HEIGHTS_HELP_STRING = (
    'List of heights (metres above ground level) for profile (vector) '
    'variables.'
)
TARGET_WAVELENGTHS_HELP_STRING = 'List of wavelengths for target variables.'
TRAIN_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The training period will be '
    '`{0:s}`...`{1:s}`.'
).format(
    FIRST_TRAIN_TIME_ARG_NAME, LAST_TRAIN_TIME_ARG_NAME
)
VALIDN_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The validation (monitoring) period '
    'will be `{0:s}`...`{1:s}`.'
).format(
    FIRST_VALIDN_TIME_ARG_NAME, LAST_VALIDN_TIME_ARG_NAME
)

NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_params`).  If you do not want to normalize, make this '
    'an empty string ("").'
)
NORMALIZE_PREDICTORS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize predictor variables.'
)
NORMALIZE_SCALAR_TARGETS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize scalar target variables.'
)
NORMALIZE_VECTOR_TARGETS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize vector target variables.'
)

NORMALIZATION_FILE_FOR_MASK_HELP_STRING = (
    'Climo-max heating rates and fluxes will be found in this file, to be read '
    'by `example_io.read_file`.  If you do not want to apply masking, leave '
    'this alone.'
)
MIN_HEATING_RATE_FOR_MASK_HELP_STRING = (
    'Minimum heating rate for masking.  Every height/wavelength pair with a '
    'climo-max heating rate below this threshold will be masked out, i.e., the '
    'NN will be forced to predict zero for this height/wavelength pair.  If '
    'you do not want to apply masking, leave this alone.'
)
MIN_FLUX_FOR_MASK_HELP_STRING = (
    'Minimum flux for masking.  Every variable/wavelength pair with a climo-'
    'max flux below this threshold will be masked out, i.e., the NN will be '
    'forced to predict zero for this variable/wavelength pair.  If you do not '
    'want to apply masking, leave this alone.'
)

BATCH_SIZE_HELP_STRING = (
    'Number of examples in each training and validation (monitoring) batch.'
)
NUM_EPOCHS_HELP_STRING = 'Number of epochs.'
NUM_TRAINING_BATCHES_HELP_STRING = 'Number of training batches per epoch.'
NUM_VALIDN_BATCHES_HELP_STRING = (
    'Number of validation (monitoring) batches per epoch.'
)
PLATEAU_LR_MULTIPLIER_HELP_STRING = (
    'Multiplier for learning rate.  Learning rate will be multiplied by this '
    'factor upon plateau in validation performance.'
)
EARLY_STOPPING_PATIENCE_HELP_STRING = (
    'Patience for early stopping.  Early stopping will be triggered if '
    'validation loss has not improved over this many epochs.'
)


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
        help=TRAINING_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
        help=VALIDATION_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + INPUT_MODEL_FILE_ARG_NAME, type=str, required=True,
        help=INPUT_MODEL_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + OUTPUT_MODEL_DIR_ARG_NAME, type=str, required=True,
        help=OUTPUT_MODEL_DIR_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_GENERATOR_FOR_TRAIN_ARG_NAME, type=int, required=False,
        default=0, help=USE_GENERATOR_FOR_TRAIN_HELP_STRING
    )
    parser_object.add_argument(
        '--' + USE_GENERATOR_FOR_VALIDN_ARG_NAME, type=int, required=False,
        default=0, help=USE_GENERATOR_FOR_VALIDN_HELP_STRING
    )
    parser_object.add_argument(
        '--' + JOINED_OUTPUT_LAYER_ARG_NAME, type=int, required=False,
        default=0, help=JOINED_OUTPUT_LAYER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_DEEP_SUPER_LAYERS_ARG_NAME, type=int, required=False,
        default=0, help=NUM_DEEP_SUPER_LAYERS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PREDICTOR_NAMES_ARG_NAME, type=str, nargs='+', required=True,
        help=PREDICTOR_NAMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_NAMES_ARG_NAME, type=str, nargs='+', required=True,
        help=TARGET_NAMES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + HEIGHTS_ARG_NAME, type=float, nargs='+', required=False,
        default=[-1], help=HEIGHTS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + TARGET_WAVELENGTHS_ARG_NAME, type=float, nargs='+',
        required=False,
        default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
        help=TARGET_WAVELENGTHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + FIRST_TRAIN_TIME_ARG_NAME, type=str, required=False,
        default='2017-01-01-000000', help=TRAIN_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAST_TRAIN_TIME_ARG_NAME, type=str, required=False,
        default='2018-12-24-235959', help=TRAIN_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + FIRST_VALIDN_TIME_ARG_NAME, type=str, required=False,
        default='2019-01-01-000000', help=VALIDN_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + LAST_VALIDN_TIME_ARG_NAME, type=str, required=False,
        default='2019-12-24-235959', help=VALIDN_TIME_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
        help=NORMALIZATION_FILE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZE_PREDICTORS_ARG_NAME, type=int, required=True,
        help=NORMALIZE_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZE_SCALAR_TARGETS_ARG_NAME, type=int, required=True,
        help=NORMALIZE_SCALAR_TARGETS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZE_VECTOR_TARGETS_ARG_NAME, type=int, required=True,
        help=NORMALIZE_VECTOR_TARGETS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NORMALIZATION_FILE_FOR_MASK_ARG_NAME, type=str, required=False,
        default='', help=NORMALIZATION_FILE_FOR_MASK_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MIN_HEATING_RATE_FOR_MASK_ARG_NAME, type=float, required=False,
        default=-1., help=MIN_HEATING_RATE_FOR_MASK_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MIN_FLUX_FOR_MASK_ARG_NAME, type=float, required=False,
        default=-1., help=MIN_FLUX_FOR_MASK_HELP_STRING
    )
    parser_object.add_argument(
        '--' + BATCH_SIZE_ARG_NAME, type=int, required=False, default=1024,
        help=BATCH_SIZE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_EPOCHS_ARG_NAME, type=int, required=False, default=1000,
        help=NUM_EPOCHS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=False,
        default=32, help=NUM_TRAINING_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=False,
        default=16, help=NUM_VALIDN_BATCHES_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PLATEAU_LR_MULTIPLIER_ARG_NAME, type=float, required=False,
        default=0.5, help=PLATEAU_LR_MULTIPLIER_HELP_STRING
    )
    parser_object.add_argument(
        '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=False,
        default=200, help=EARLY_STOPPING_PATIENCE_HELP_STRING
    )

    return parser_object
