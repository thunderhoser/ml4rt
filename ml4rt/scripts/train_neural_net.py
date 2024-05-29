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


def _run(training_dir_name, validation_dir_name,
         input_model_file_name, output_model_dir_name,
         use_generator_for_training, use_generator_for_validn,
         joined_output_layer, num_deep_supervision_layers, predictor_names,
         target_names, heights_m_agl, target_wavelengths_metres,
         first_training_time_string, last_training_time_string,
         first_validn_time_string, last_validn_time_string,
         normalization_file_name, normalize_predictors,
         normalize_scalar_targets, normalize_vector_targets,
         normalization_file_name_for_mask, min_heating_rate_for_mask_k_day01,
         min_flux_for_mask_w_m02,
         num_examples_per_batch, num_epochs, num_training_batches_per_epoch,
         num_validn_batches_per_epoch, plateau_lr_multiplier):
    """Trains neural net

    :param training_dir_name: See documentation at top of training_args.py.
    :param validation_dir_name: Same.
    :param input_model_file_name: Same.
    :param output_model_dir_name: Same.
    :param use_generator_for_training: Same.
    :param use_generator_for_validn: Same.
    :param joined_output_layer: Same.
    :param num_deep_supervision_layers: Same.
    :param predictor_names: Same.
    :param target_names: Same.
    :param heights_m_agl: Same.
    :param target_wavelengths_metres: Same.
    :param first_training_time_string: Same.
    :param last_training_time_string: Same.
    :param first_validn_time_string: Same.
    :param last_validn_time_string: Same.
    :param normalization_file_name: Same.
    :param normalize_predictors: Same.
    :param normalize_scalar_targets: Same.
    :param normalize_vector_targets: Same.
    :param normalization_file_name_for_mask: Same.
    :param min_heating_rate_for_mask_k_day01: Same.
    :param min_flux_for_mask_w_m02: Same.
    :param num_examples_per_batch: Same.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param num_validn_batches_per_epoch: Same.
    :param plateau_lr_multiplier: Same.
    """

    if normalization_file_name in NONE_STRINGS:
        normalization_file_name = None

    if (
            normalization_file_name_for_mask in NONE_STRINGS
            or min_heating_rate_for_mask_k_day01 <= 0
            or min_flux_for_mask_w_m02 <= 0
    ):
        normalization_file_name_for_mask = None
        min_heating_rate_for_mask_k_day01 = None
        min_flux_for_mask_w_m02 = None

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
        neural_net.EXAMPLE_DIRECTORY_KEY: training_dir_name,
        neural_net.BATCH_SIZE_KEY: num_examples_per_batch,
        neural_net.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        neural_net.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        neural_net.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        neural_net.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        neural_net.HEIGHTS_KEY: heights_m_agl,
        neural_net.TARGET_WAVELENGTHS_KEY: target_wavelengths_metres,
        neural_net.NORMALIZATION_FILE_KEY: normalization_file_name,
        neural_net.NORMALIZE_PREDICTORS_KEY: normalize_predictors,
        neural_net.NORMALIZE_SCALAR_TARGETS_KEY: normalize_scalar_targets,
        neural_net.NORMALIZE_VECTOR_TARGETS_KEY: normalize_vector_targets,
        neural_net.FIRST_TIME_KEY: first_training_time_unix_sec,
        neural_net.LAST_TIME_KEY: last_training_time_unix_sec,
        neural_net.JOINED_OUTPUT_LAYER_KEY: joined_output_layer,
        neural_net.NUM_DEEP_SUPER_LAYERS_KEY: num_deep_supervision_layers,
        neural_net.NORMALIZATION_FILE_FOR_MASK_KEY:
            normalization_file_name_for_mask,
        neural_net.MIN_HEATING_RATE_FOR_MASK_KEY:
            min_heating_rate_for_mask_k_day01,
        neural_net.MIN_FLUX_FOR_MASK_KEY: min_flux_for_mask_w_m02
    }

    validation_option_dict = {
        neural_net.EXAMPLE_DIRECTORY_KEY: validation_dir_name,
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

    print('Reading loss functions and architecture from: "{0:s}"...'.format(
        input_metafile_name
    ))
    metadata_dict = neural_net.read_metafile(input_metafile_name)
    loss_function_or_dict = metadata_dict[neural_net.LOSS_FUNCTION_OR_DICT_KEY]
    dense_architecture_dict = metadata_dict[neural_net.DENSE_ARCHITECTURE_KEY]
    cnn_architecture_dict = metadata_dict[neural_net.CNN_ARCHITECTURE_KEY]
    bnn_architecture_dict = metadata_dict[neural_net.BNN_ARCHITECTURE_KEY]
    u_net_architecture_dict = metadata_dict[neural_net.U_NET_ARCHITECTURE_KEY]
    u_net_plusplus_architecture_dict = metadata_dict[
        neural_net.U_NET_PP_ARCHITECTURE_KEY
    ]
    u_net_3plus_architecture_dict = metadata_dict[
        neural_net.U_NET_PPP_ARCHITECTURE_KEY
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
            loss_function_or_dict=loss_function_or_dict, do_early_stopping=True,
            plateau_lr_multiplier=plateau_lr_multiplier,
            dense_architecture_dict=dense_architecture_dict,
            cnn_architecture_dict=cnn_architecture_dict,
            bnn_architecture_dict=bnn_architecture_dict,
            u_net_architecture_dict=u_net_architecture_dict,
            u_net_plusplus_architecture_dict=u_net_plusplus_architecture_dict,
            u_net_3plus_architecture_dict=u_net_3plus_architecture_dict
        )
    else:
        neural_net.train_model_sans_generator(
            model_object=model_object, output_dir_name=output_model_dir_name,
            num_epochs=num_epochs, training_option_dict=training_option_dict,
            validation_option_dict=validation_option_dict,
            loss_function_or_dict=loss_function_or_dict, do_early_stopping=True,
            num_training_batches_per_epoch=num_training_batches_per_epoch,
            num_validation_batches_per_epoch=num_validn_batches_per_epoch,
            plateau_lr_multiplier=plateau_lr_multiplier,
            dense_architecture_dict=dense_architecture_dict,
            cnn_architecture_dict=cnn_architecture_dict,
            bnn_architecture_dict=bnn_architecture_dict,
            u_net_architecture_dict=u_net_architecture_dict,
            u_net_plusplus_architecture_dict=u_net_plusplus_architecture_dict,
            u_net_3plus_architecture_dict=u_net_3plus_architecture_dict
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        training_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.TRAINING_DIR_ARG_NAME
        ),
        validation_dir_name=getattr(
            INPUT_ARG_OBJECT, training_args.VALIDATION_DIR_ARG_NAME
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
        joined_output_layer=bool(getattr(
            INPUT_ARG_OBJECT, training_args.JOINED_OUTPUT_LAYER_ARG_NAME
        )),
        num_deep_supervision_layers=getattr(
            INPUT_ARG_OBJECT, training_args.NUM_DEEP_SUPER_LAYERS_ARG_NAME
        ),
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
        target_wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, training_args.TARGET_WAVELENGTHS_ARG_NAME),
            dtype=float
        ),
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
        normalize_predictors=bool(getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZE_PREDICTORS_ARG_NAME
        )),
        normalize_scalar_targets=bool(getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZE_SCALAR_TARGETS_ARG_NAME
        )),
        normalize_vector_targets=bool(getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZE_VECTOR_TARGETS_ARG_NAME
        )),
        normalization_file_name_for_mask=getattr(
            INPUT_ARG_OBJECT, training_args.NORMALIZATION_FILE_FOR_MASK_ARG_NAME
        ),
        min_heating_rate_for_mask_k_day01=getattr(
            INPUT_ARG_OBJECT, training_args.MIN_HEATING_RATE_FOR_MASK_ARG_NAME
        ),
        min_flux_for_mask_w_m02=getattr(
            INPUT_ARG_OBJECT, training_args.MIN_FLUX_FOR_MASK_ARG_NAME
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
        ),
        plateau_lr_multiplier=getattr(
            INPUT_ARG_OBJECT, training_args.PLATEAU_LR_MULTIPLIER_ARG_NAME
        )
    )
