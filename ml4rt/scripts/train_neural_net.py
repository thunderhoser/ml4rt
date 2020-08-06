"""Trains neural net."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net
from ml4rt.scripts import training_args

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NONE_STRINGS = ['', 'none', 'None']

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER = training_args.add_input_args(parser_object=INPUT_ARG_PARSER)


def _run(net_type_string, example_dir_name, input_model_file_name,
         output_model_dir_name,
         use_generator_for_training, use_generator_for_validn,
         predictor_names, target_names, heights_m_agl,
         omit_heating_rate, first_training_time_string,
         last_training_time_string, first_validn_time_string,
         last_validn_time_string, normalization_file_name,
         predictor_norm_type_string, predictor_min_norm_value,
         predictor_max_norm_value, vector_target_norm_type_string,
         vector_target_min_norm_value, vector_target_max_norm_value,
         scalar_target_norm_type_string, scalar_target_min_norm_value,
         scalar_target_max_norm_value, num_examples_per_batch,
         num_epochs, num_training_batches_per_epoch,
         num_validn_batches_per_epoch):
    """Trains neural net

    :param net_type_string: See documentation at top of training_args.py.
    :param example_dir_name: Same.
    :param input_model_file_name: Same.
    :param output_model_dir_name: Same.
    :param use_generator_for_training: Same.
    :param use_generator_for_validn: Same.
    :param predictor_names: Same.
    :param target_names: Same.
    :param heights_m_agl: Same.
    :param omit_heating_rate: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param normalization_file_name: Same.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param vector_target_norm_type_string: Same.
    :param vector_target_min_norm_value: Same.
    :param vector_target_max_norm_value: Same.
    :param scalar_target_norm_type_string: Same.
    :param scalar_target_min_norm_value: Same.
    :param scalar_target_max_norm_value: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validn_batches_per_epoch: Same.
    """

    if predictor_norm_type_string in NONE_STRINGS:
        predictor_norm_type_string = None
    if vector_target_norm_type_string in NONE_STRINGS:
        vector_target_norm_type_string = None
    if scalar_target_norm_type_string in NONE_STRINGS:
        scalar_target_norm_type_string = None

    neural_net.check_net_type(net_type_string)

    if len(heights_m_agl) and heights_m_agl[0] <= 0:
        heights_m_agl = (
            training_args.NET_TYPE_TO_DEFAULT_HEIGHTS_M_AGL[net_type_string]
        )

    for n in predictor_names:
        example_utils.check_field_name(n)

    scalar_predictor_names = [
        n for n in predictor_names
        if n in example_utils.ALL_SCALAR_PREDICTOR_NAMES
    ]
    vector_predictor_names = [
        n for n in predictor_names
        if n in example_utils.ALL_VECTOR_PREDICTOR_NAMES
    ]

    for n in target_names:
        example_utils.check_field_name(n)

    scalar_target_names = [
        n for n in target_names if n in example_utils.ALL_SCALAR_TARGET_NAMES
    ]
    vector_target_names = [
        n for n in target_names if n in example_utils.ALL_VECTOR_TARGET_NAMES
    ]

    first_training_time_unix_sec = time_conversion.string_to_unix_sec(
        first_training_time_string, training_args.TIME_FORMAT
    )
    last_training_time_unix_sec = time_conversion.string_to_unix_sec(
        last_training_time_string, training_args.TIME_FORMAT
    )
    first_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        first_validn_time_string, training_args.TIME_FORMAT
    )
    last_validn_time_unix_sec = time_conversion.string_to_unix_sec(
        last_validn_time_string, training_args.TIME_FORMAT
    )

    training_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: example_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        neural_net.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        neural_net.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        neural_net.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        neural_net.HEIGHTS_KEY: heights_m_agl,
        neural_net.OMIT_HEATING_RATE_KEY: omit_heating_rate,
        neural_net.NORMALIZATION_FILE_KEY: normalization_file_name,
        neural_net.PREDICTOR_NORM_TYPE_KEY: predictor_norm_type_string,
        neural_net.PREDICTOR_MIN_NORM_VALUE_KEY: predictor_min_norm_value,
        neural_net.PREDICTOR_MAX_NORM_VALUE_KEY: predictor_max_norm_value,
        neural_net.VECTOR_TARGET_NORM_TYPE_KEY: vector_target_norm_type_string,
        neural_net.VECTOR_TARGET_MIN_VALUE_KEY: vector_target_min_norm_value,
        neural_net.VECTOR_TARGET_MAX_VALUE_KEY: vector_target_max_norm_value,
        neural_net.SCALAR_TARGET_NORM_TYPE_KEY: scalar_target_norm_type_string,
        neural_net.SCALAR_TARGET_MIN_VALUE_KEY: scalar_target_min_norm_value,
        neural_net.SCALAR_TARGET_MAX_VALUE_KEY: scalar_target_max_norm_value,
        neural_net.FIRST_TIME_KEY: first_training_time_unix_sec,
        neural_net.LAST_TIME_KEY: last_training_time_unix_sec,
        # neural_net.MIN_COLUMN_LWP_KEY: 0.05,
        # neural_net.MAX_COLUMN_LWP_KEY: 1e12
    }

    validation_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: example_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.FIRST_TIME_KEY: first_validn_time_unix_sec,
        neural_net.LAST_TIME_KEY: last_validn_time_unix_sec
    }

    print('Reading untrained model from: "{0:s}"...'.format(
        input_model_file_name
    ))
    model_object = neural_net.read_model(input_model_file_name)

    input_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(input_model_file_name)[0]
    )

    print('Reading loss function from: "{0:s}"...'.format(input_metafile_name))
    loss_function = neural_net.read_metafile(input_metafile_name)[
        neural_net.LOSS_FUNCTION_KEY
    ]

    print(SEPARATOR_STRING)

    if use_generator_for_training:
        neural_net.train_model_with_generator(
            model_object=model_object, output_dir_name=output_model_dir_name,
            num_epochs=num_epochs,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            training_option_dict=training_option_dict,
            use_generator_for_validn=use_generator_for_validn,
            num_validation_batches_per_epoch=num_validn_batches_per_epoch,
            validation_option_dict=validation_option_dict,
            net_type_string=net_type_string,
            loss_function=loss_function, do_early_stopping=True
        )
    else:
        neural_net.train_model_sans_generator(
            model_object=model_object, output_dir_name=output_model_dir_name,
            num_epochs=num_epochs,
            training_option_dict=training_option_dict,
            validation_option_dict=validation_option_dict,
            net_type_string=net_type_string,
            loss_function=loss_function, do_early_stopping=True
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        net_type_string=getattr(
            INPUT_ARG_OBJECT, training_args.NET_TYPE_ARG_NAME
        ),
        example_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.EXAMPLE_DIR_ARG_NAME
        ),
        input_model_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.INPUT_MODEL_FILE_ARG_NAME
        ),
        output_model_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.OUTPUT_MODEL_DIR_ARG_NAME
        ),
        use_generator_for_training=bool(getattr(
            INPUT_ARG_OBJECT, training_args.USE_GENERATOR_FOR_TRAIN_ARG_NAME
        )),
        use_generator_for_validn=bool(getattr(
            INPUT_ARG_OBJECT, training_args.USE_GENERATOR_FOR_VALIDN_ARG_NAME
        )),
        predictor_names=getattr(
            INPUT_ARG_OBJECT, training_args.PREDICTOR_NAMES_ARG_NAME
        ),
        target_names=getattr(
            INPUT_ARG_OBJECT, training_args.TARGET_NAMES_ARG_NAME
        ),
        heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.HEIGHTS_ARG_NAME),
            dtype=float
        ),
        omit_heating_rate=bool(getattr(
            INPUT_ARG_OBJECT, training_args.OMIT_HEATING_RATE_ARG_NAME
        )),
        first_training_time_string=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_TRAIN_TIME_ARG_NAME
        ),
        last_training_time_string=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_TRAIN_TIME_ARG_NAME
        ),
        first_validn_time_string=getattr(
            INPUT_ARG_OBJECT, training_args.FIRST_VALIDN_TIME_ARG_NAME
        ),
        last_validn_time_string=getattr(
            INPUT_ARG_OBJECT, training_args.LAST_VALIDN_TIME_ARG_NAME
        ),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZATION_FILE_ARG_NAME
        ),
        predictor_norm_type_string=getattr(
            INPUT_ARG_OBJECT, training_args.PREDICTOR_NORM_TYPE_ARG_NAME
        ),
        predictor_min_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.PREDICTOR_MIN_VALUE_ARG_NAME
        ),
        predictor_max_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.PREDICTOR_MAX_VALUE_ARG_NAME
        ),
        vector_target_norm_type_string=getattr(
            INPUT_ARG_OBJECT, training_args.VECTOR_TARGET_NORM_TYPE_ARG_NAME
        ),
        vector_target_min_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.VECTOR_TARGET_MIN_VALUE_ARG_NAME
        ),
        vector_target_max_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.VECTOR_TARGET_MAX_VALUE_ARG_NAME
        ),
        scalar_target_norm_type_string=getattr(
            INPUT_ARG_OBJECT, training_args.SCALAR_TARGET_NORM_TYPE_ARG_NAME
        ),
        scalar_target_min_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.SCALAR_TARGET_MIN_VALUE_ARG_NAME
        ),
        scalar_target_max_norm_value=getattr(
            INPUT_ARG_OBJECT, training_args.SCALAR_TARGET_MAX_VALUE_ARG_NAME
        ),
        num_examples_per_batch=getattr(
            INPUT_ARG_OBJECT, training_args.BATCH_SIZE_ARG_NAME
        ),
        num_epochs=getattr(INPUT_ARG_OBJECT, training_args.NUM_EPOCHS_ARG_NAME),
        num_training_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_TRAINING_BATCHES_ARG_NAME
        ),
        num_validn_batches_per_epoch=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_VALIDN_BATCHES_ARG_NAME
        )
    )
