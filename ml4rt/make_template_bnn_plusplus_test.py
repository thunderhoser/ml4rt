"""Makes test template for Bayesian U-net++."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_pp_architecture_bayesian as u_net_pp_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/bnn_plusplus_test/template'
).format(HOME_DIR_NAME)

MODEL_DEPTH = 3
VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_crps()
SCALAR_LOSS_FUNCTION = custom_losses.unscaled_crps_for_net_flux()
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_crps()',
    'dense_output': 'custom_losses.unscaled_crps_for_net_flux()'
}

NUM_HEIGHTS_AT_DEEPEST_LAYER = 15
NUM_CHANNELS_AT_DEEPEST_LAYER = 1024
DUMMY_TRAINING_SIZE = int(1e5)

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=(
        NUM_HEIGHTS_AT_DEEPEST_LAYER * NUM_CHANNELS_AT_DEEPEST_LAYER
    ),
    num_classes=2, num_dense_layers=4, for_classification=False
)[1]

POINT_ESTIMATE_TYPE_STRING = u_net_pp_architecture.POINT_ESTIMATE_TYPE_STRING
FLIPOUT_TYPE_STRING = u_net_pp_architecture.FLIPOUT_TYPE_STRING
REPARAMETERIZATION_TYPE_STRING = (
    u_net_pp_architecture.REPARAMETERIZATION_TYPE_STRING
)

OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: MODEL_DEPTH,
    u_net_pp_architecture.KL_SCALING_FACTOR_KEY: DUMMY_TRAINING_SIZE ** -1,
    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.full(MODEL_DEPTH + 1, 1, dtype=int),
    u_net_pp_architecture.CHANNEL_COUNTS_KEY:
        numpy.array([128, 256, 512, 1024], dtype=int),
    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH + 1, 0.),
    u_net_pp_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH + 1, False, dtype=bool),
    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, False, dtype=bool),
    u_net_pp_architecture.UPCONV_BNN_LAYER_TYPES_KEY: [
        POINT_ESTIMATE_TYPE_STRING, POINT_ESTIMATE_TYPE_STRING,
        FLIPOUT_TYPE_STRING
    ],
    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_architecture.SKIP_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, False, dtype=bool),
    u_net_pp_architecture.SKIP_BNN_LAYER_TYPES_KEY: [
        POINT_ESTIMATE_TYPE_STRING, POINT_ESTIMATE_TYPE_STRING,
        FLIPOUT_TYPE_STRING
    ],
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: False,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_pp_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_architecture.L1_WEIGHT_KEY: 0.,
    u_net_pp_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True,
    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
        DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(4, False, dtype=bool),
    u_net_pp_architecture.DENSE_BNN_LAYER_TYPES_KEY: [
        POINT_ESTIMATE_TYPE_STRING, POINT_ESTIMATE_TYPE_STRING,
        FLIPOUT_TYPE_STRING, FLIPOUT_TYPE_STRING
    ],
    u_net_pp_architecture.CONV_OUTPUT_BNN_LAYER_TYPE_KEY: FLIPOUT_TYPE_STRING
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes test template for Bayesian U-net++.

    This is effectively the main method.
    """

    this_model_object = u_net_pp_architecture.create_bayesian_model(
        option_dict=OPTION_DICT,
        vector_loss_function=VECTOR_LOSS_FUNCTION,
        num_output_channels=1,
        scalar_loss_function=SCALAR_LOSS_FUNCTION
    )

    this_model_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(
        file_name=this_model_file_name
    )

    print('Writing model to: "{0:s}"...'.format(this_model_file_name))
    this_model_object.save(
        filepath=this_model_file_name, overwrite=True,
        include_optimizer=True
    )

    this_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(this_model_file_name)[0],
        raise_error_if_missing=False
    )

    print('Writing metadata to: "{0:s}"...'.format(this_metafile_name))
    neural_net._write_metafile(
        dill_file_name=this_metafile_name, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        num_validation_batches_per_epoch=100,
        validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        net_type_string=neural_net.U_NET_TYPE_STRING,
        loss_function_or_dict=LOSS_DICT,
        do_early_stopping=True, plateau_lr_multiplier=0.6
    )


if __name__ == '__main__':
    _run()
