"""Contains list of input arguments for training a neural net."""

from ml4rt.utils import normalization

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

NET_TYPE_ARG_NAME = 'net_type_string'
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
MULTIPLY_PREDICTORS_ARG_NAME = 'multiply_preds_by_layer_thickness'
MULTIPLY_HEATING_RATES_ARG_NAME = 'multiply_hr_by_layer_thickness'
FIRST_TRAIN_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAIN_TIME_ARG_NAME = 'last_training_time_string'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
PREDICTOR_NORM_TYPE_ARG_NAME = 'predictor_norm_type_string'
PREDICTOR_MIN_VALUE_ARG_NAME = 'predictor_min_norm_value'
PREDICTOR_MAX_VALUE_ARG_NAME = 'predictor_max_norm_value'
VECTOR_TARGET_NORM_TYPE_ARG_NAME = 'vector_target_norm_type_string'
VECTOR_TARGET_MIN_VALUE_ARG_NAME = 'vector_target_min_norm_value'
VECTOR_TARGET_MAX_VALUE_ARG_NAME = 'vector_target_max_norm_value'
SCALAR_TARGET_NORM_TYPE_ARG_NAME = 'scalar_target_norm_type_string'
SCALAR_TARGET_MIN_VALUE_ARG_NAME = 'scalar_target_min_norm_value'
SCALAR_TARGET_MAX_VALUE_ARG_NAME = 'scalar_target_max_norm_value'
BATCH_SIZE_ARG_NAME = 'num_examples_per_batch'
NUM_EPOCHS_ARG_NAME = 'num_epochs'
NUM_TRAINING_BATCHES_ARG_NAME = 'num_training_batches_per_epoch'
NUM_VALIDN_BATCHES_ARG_NAME = 'num_validn_batches_per_epoch'
PLATEAU_LR_MULTIPLIER_ARG_NAME = 'plateau_lr_multiplier'

NET_TYPE_HELP_STRING = (
    'Neural-net type (must be accepted by `neural_net.check_net_type`).'
)
TRAINING_DIR_HELP_STRING = (
    'Name of directory with training examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
VALIDATION_DIR_HELP_STRING = (
    'Same as `{0:s}` but for validation (monitoring) examples.'
).format(TRAINING_DIR_ARG_NAME)

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
).format(USE_GENERATOR_FOR_TRAIN_ARG_NAME)

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
MULTIPLY_PREDICTORS_HELP_STRING = (
    'Boolean flag.  If 1, will multiply relevant predictors by layer thickness.'
)
MULTIPLY_HEATING_RATES_HELP_STRING = (
    'Boolean flag.  If 1, will multiply heating rates by layer thickness.'
)
TRAIN_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The training period will be '
    '`{0:s}`...`{1:s}`.'
).format(FIRST_TRAIN_TIME_ARG_NAME, LAST_TRAIN_TIME_ARG_NAME)

VALIDN_TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The validation (monitoring) period '
    'will be `{0:s}`...`{1:s}`.'
).format(FIRST_VALIDN_TIME_ARG_NAME, LAST_VALIDN_TIME_ARG_NAME)

NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  Will be read by `example_io.read_file`.  If '
    'you do not want to normalize, make this an empty string ("").'
)
PREDICTOR_NORM_TYPE_HELP_STRING = (
    'Normalization type for predictors (must be accepted by '
    '`normalization.check_normalization_type`).  If you do not want to '
    'normalize, make this an empty string ("").'
)
PREDICTOR_MIN_VALUE_HELP_STRING = (
    'Minimum value if you have chosen min-max normalization.'
)
PREDICTOR_MAX_VALUE_HELP_STRING = (
    'Max value if you have chosen min-max normalization.'
)
VECTOR_TARGET_NORM_TYPE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_NORM_TYPE_ARG_NAME)

VECTOR_TARGET_MIN_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_MIN_VALUE_ARG_NAME)

VECTOR_TARGET_MAX_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_MAX_VALUE_ARG_NAME)

SCALAR_TARGET_NORM_TYPE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_NORM_TYPE_ARG_NAME)

SCALAR_TARGET_MIN_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_MIN_VALUE_ARG_NAME)

SCALAR_TARGET_MAX_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_MAX_VALUE_ARG_NAME)

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


def add_input_args(parser_object):
    """Adds input args to ArgumentParser object.

    :param parser_object: Instance of `argparse.ArgumentParser` (may already
        contain some input args).
    :return: parser_object: Same as input but with new args added.
    """

    parser_object.add_argument(
        '--' + NET_TYPE_ARG_NAME, type=str, required=True,
        help=NET_TYPE_HELP_STRING
    )
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
        '--' + MULTIPLY_PREDICTORS_ARG_NAME, type=int, required=False,
        default=0, help=MULTIPLY_PREDICTORS_HELP_STRING
    )
    parser_object.add_argument(
        '--' + MULTIPLY_HEATING_RATES_ARG_NAME, type=int, required=False,
        default=0, help=MULTIPLY_HEATING_RATES_HELP_STRING
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
        '--' + PREDICTOR_NORM_TYPE_ARG_NAME, type=str, required=False,
        default=normalization.Z_SCORE_NORM_STRING,
        help=PREDICTOR_NORM_TYPE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PREDICTOR_MIN_VALUE_ARG_NAME, type=float, required=False,
        default=0., help=PREDICTOR_MIN_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + PREDICTOR_MAX_VALUE_ARG_NAME, type=float, required=False,
        default=1., help=PREDICTOR_MAX_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VECTOR_TARGET_NORM_TYPE_ARG_NAME, type=str, required=True,
        help=VECTOR_TARGET_NORM_TYPE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VECTOR_TARGET_MIN_VALUE_ARG_NAME, type=float, required=False,
        default=0., help=VECTOR_TARGET_MIN_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + VECTOR_TARGET_MAX_VALUE_ARG_NAME, type=float, required=False,
        default=1., help=VECTOR_TARGET_MAX_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SCALAR_TARGET_NORM_TYPE_ARG_NAME, type=str, required=True,
        help=SCALAR_TARGET_NORM_TYPE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SCALAR_TARGET_MIN_VALUE_ARG_NAME, type=float, required=False,
        default=0., help=SCALAR_TARGET_MIN_VALUE_HELP_STRING
    )
    parser_object.add_argument(
        '--' + SCALAR_TARGET_MAX_VALUE_ARG_NAME, type=float, required=False,
        default=1., help=SCALAR_TARGET_MAX_VALUE_HELP_STRING
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

    return parser_object
