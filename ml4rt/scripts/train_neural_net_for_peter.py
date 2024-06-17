"""Trains neural net for Peter Ukkonen."""

import argparse
from gewittergefahr.gg_utils import time_conversion
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import peter_brnn_architecture
from ml4rt.machine_learning import peter_brnn_architecture_ryan

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

TRAINING_DIR_ARG_NAME = 'input_training_dir_name'
VALIDATION_DIR_ARG_NAME = 'input_validation_dir_name'
OUTPUT_MODEL_DIR_ARG_NAME = 'output_model_dir_name'
USE_GENERATOR_FOR_VALIDN_ARG_NAME = 'use_generator_for_validn'
USE_RYAN_ARCHITECTURE_ARG_NAME = 'use_ryan_architecture'

FIRST_TRAIN_TIME_ARG_NAME = 'first_training_time_string'
LAST_TRAIN_TIME_ARG_NAME = 'last_training_time_string'
FIRST_VALIDN_TIME_ARG_NAME = 'first_validn_time_string'
LAST_VALIDN_TIME_ARG_NAME = 'last_validn_time_string'

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
OUTPUT_MODEL_DIR_HELP_STRING = (
    'Name of output directory.  Model will be saved here.'
)
USE_GENERATOR_FOR_VALIDN_HELP_STRING = (
    'Boolean flag.  If 1, will use generator for validation data.  If 0, will '
    'load all validation data into memory at once.'
)
USE_RYAN_ARCHITECTURE_HELP_STRING = (
    'Boolean flag.  If 1, will use Ryan''s version of Peter''s architecture.'
)

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

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALIDATION_DIR_ARG_NAME, type=str, required=True,
    help=VALIDATION_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_MODEL_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_MODEL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_GENERATOR_FOR_VALIDN_ARG_NAME, type=int, required=False,
    default=1, help=USE_GENERATOR_FOR_VALIDN_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_RYAN_ARCHITECTURE_ARG_NAME, type=int, required=True,
    help=USE_RYAN_ARCHITECTURE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TRAIN_TIME_ARG_NAME, type=str, required=True,
    help=TRAIN_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TRAIN_TIME_ARG_NAME, type=str, required=True,
    help=TRAIN_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_VALIDN_TIME_ARG_NAME, type=str, required=True,
    help=VALIDN_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_VALIDN_TIME_ARG_NAME, type=str, required=True,
    help=VALIDN_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + BATCH_SIZE_ARG_NAME, type=int, required=True,
    help=BATCH_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EPOCHS_ARG_NAME, type=int, required=True,
    help=NUM_EPOCHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TRAINING_BATCHES_ARG_NAME, type=int, required=True,
    help=NUM_TRAINING_BATCHES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_VALIDN_BATCHES_ARG_NAME, type=int, required=True,
    help=NUM_VALIDN_BATCHES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLATEAU_LR_MULTIPLIER_ARG_NAME, type=float, required=True,
    help=PLATEAU_LR_MULTIPLIER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EARLY_STOPPING_PATIENCE_ARG_NAME, type=int, required=True,
    help=EARLY_STOPPING_PATIENCE_HELP_STRING
)


def _run(training_dir_name, validation_dir_name, output_model_dir_name,
         use_generator_for_validn, use_ryan_architecture,
         first_training_time_string, last_training_time_string,
         first_validn_time_string, last_validn_time_string,
         num_examples_per_batch, num_epochs,
         num_training_batches_per_epoch, num_validn_batches_per_epoch,
         plateau_lr_multiplier, early_stopping_patience_epochs):
    """Trains neural net for Peter Ukkonen.

    This is effectively the main method.

    :param training_dir_name: See documentation at top of this script.
    :param validation_dir_name: Same.
    :param output_model_dir_name: Same.
    :param use_generator_for_validn: Same.
    :param use_ryan_architecture: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validn_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, TIME_FORMAT
    )
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, TIME_FORMAT
    )
    first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validn_time_string, TIME_FORMAT
    )
    last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validn_time_string, TIME_FORMAT
    )

    training_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: training_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.FIRST_TIME_KEY: first_training_time_unix_sec,
        neural_net.LAST_TIME_KEY: last_training_time_unix_sec
    }
    validation_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: validation_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.FIRST_TIME_KEY: first_validn_time_unix_sec,
        neural_net.LAST_TIME_KEY: last_validn_time_unix_sec
    }

    if use_ryan_architecture:
        model_object = peter_brnn_architecture_ryan.rnn_sw()
    else:
        model_object = peter_brnn_architecture.rnn_sw()

    neural_net.train_model_with_generator_for_peter(
        model_object=model_object,
        output_dir_name=output_model_dir_name,
        num_epochs=num_epochs,
        training_option_dict=training_option_dict,
        validation_option_dict=validation_option_dict,
        use_generator_for_validn=use_generator_for_validn,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        num_validation_batches_per_epoch=num_validn_batches_per_epoch,
        plateau_lr_multiplier=plateau_lr_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        use_ryan_architecture=use_ryan_architecture
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        training_dir_name=getattr(INPUT_ARG_OBJECT, TRAINING_DIR_ARG_NAME),
        validation_dir_name=getattr(INPUT_ARG_OBJECT, VALIDATION_DIR_ARG_NAME),
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_MODEL_DIR_ARG_NAME
        ),
        use_generator_for_validn=bool(getattr(
            INPUT_ARG_OBJECT, USE_GENERATOR_FOR_VALIDN_ARG_NAME
        )),
        use_ryan_architecture=bool(getattr(
            INPUT_ARG_OBJECT, USE_RYAN_ARCHITECTURE_ARG_NAME
        )),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_TRAIN_TIME_ARG_NAME
        ),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_TRAIN_TIME_ARG_NAME
        ),
        first_validn_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_VALIDN_TIME_ARG_NAME
        ),
        last_validn_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_VALIDN_TIME_ARG_NAME
        ),
        num_examples_per_batch=getattr(INPUT_ARG_OBJECT, BATCH_SIZE_ARG_NAME),
        num_epochs=getattr(INPUT_ARG_OBJECT, NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validn_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, NUM_VALIDN_BATCHES_ARG_NAME
        ),
        plateau_lr_multiplier=getattr(
            INPUT_ARG_OBJECT, PLATEAU_LR_MULTIPLIER_ARG_NAME
        ),
        early_stopping_patience_epochs=getattr(
            INPUT_ARG_OBJECT, EARLY_STOPPING_PATIENCE_ARG_NAME
        )
    )
