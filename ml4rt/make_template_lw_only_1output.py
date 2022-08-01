"""Makes template for experiment with longwave only, one joined output layer."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_pp_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4rt_models/lw_only_1output/template'.format(
    HOME_DIR_NAME
)

NUM_LEVELS = 4
FLUX_LOSS_SCALING_FACTOR = 0.5
LOSS_FUNCTION = custom_losses.joined_output_loss(127, 0.5)
LOSS_FUNCTION_STRING = 'custom_losses.joined_output_loss(127, 0.5)'

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=7 * 1024, num_classes=2, num_dense_layers=4,
    for_classification=False
)[1]

DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 23], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    u_net_pp_architecture.CHANNEL_COUNTS_KEY: numpy.round(
        numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    ).astype(int),
    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS + 1, 0.),
    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_pp_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_pp_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY: None,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_pp_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_pp_architecture.L1_WEIGHT_KEY: 0.,
    u_net_pp_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True,
    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes template for experiment with longwave only, one joined output lyr.

    This is effectively the main method.
    """

    model_object = u_net_pp_architecture.create_model_1output_layer(
        option_dict=DEFAULT_OPTION_DICT, loss_function=LOSS_FUNCTION,
        num_output_channels=1
    )

    model_file_name = '{0:s}/model.h5'.format(OUTPUT_DIR_NAME)
    file_system_utils.mkdir_recursive_if_necessary(file_name=model_file_name)

    print('Writing model to: "{0:s}"...'.format(model_file_name))
    model_object.save(
        filepath=model_file_name, overwrite=True, include_optimizer=True
    )

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=False
    )

    print('Writing metadata to: "{0:s}"...'.format(metafile_name))
    neural_net._write_metafile(
        dill_file_name=metafile_name, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        num_validation_batches_per_epoch=100,
        validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        net_type_string=neural_net.U_NET_TYPE_STRING,
        loss_function_or_dict=LOSS_FUNCTION_STRING,
        do_early_stopping=True, plateau_lr_multiplier=0.6
    )


if __name__ == '__main__':
    _run()
