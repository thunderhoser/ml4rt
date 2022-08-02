"""Makes linear-regression template for Tom Beucler."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import dense_net_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/tom_experiment/linear_regression/templates'
).format(HOME_DIR_NAME)

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(0.64)
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_mse()',
    'dense_output': 'custom_losses.scaled_mse_for_net_flux(0.64)'
}

DEFAULT_OPTION_DICT = {
    dense_net_architecture.NUM_INPUTS_KEY: 127 * 22 + 4,
    dense_net_architecture.NUM_HEIGHTS_KEY: 127,
    dense_net_architecture.NUM_FLUX_COMPONENTS_KEY: 2,
    dense_net_architecture.HIDDEN_LAYER_NEURON_NUMS_KEY:
        numpy.array([], dtype=int),
    dense_net_architecture.HIDDEN_LAYER_DROPOUT_RATES_KEY:
        numpy.array([], dtype=float),
    dense_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    dense_net_architecture.HEATING_RATE_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.HEATING_RATE_ACTIV_FUNC_ALPHA_KEY: 0.,
    dense_net_architecture.FLUX_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    dense_net_architecture.FLUX_ACTIV_FUNC_ALPHA_KEY: 0.,
    dense_net_architecture.L1_WEIGHT_KEY: 0.,
    dense_net_architecture.L2_WEIGHT_KEY: 1e-7,
    dense_net_architecture.USE_BATCH_NORM_KEY: True
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes linear-regression template for Tom Beucler.

    This is effectively the main method.
    """

    model_object = dense_net_architecture.create_model(
        option_dict=DEFAULT_OPTION_DICT,
        heating_rate_loss_function=VECTOR_LOSS_FUNCTION,
        flux_loss_function=SCALAR_LOSS_FUNCTION
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

    print('Writing metadata to: "{0:s}"...'.format(
        metafile_name
    ))
    neural_net._write_metafile(
        dill_file_name=metafile_name, num_epochs=100,
        num_training_batches_per_epoch=100,
        training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        num_validation_batches_per_epoch=100,
        validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
        net_type_string=neural_net.DENSE_NET_TYPE_STRING,
        loss_function_or_dict=LOSS_DICT,
        do_early_stopping=True, plateau_lr_multiplier=0.6
    )


if __name__ == '__main__':
    _run()
