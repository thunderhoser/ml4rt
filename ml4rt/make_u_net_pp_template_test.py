"""Makes U-net ++ template for prelim test."""

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
import normalization
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = '{0:s}/ml4rt_models/u-net++_test/template'.format(
    HOME_DIR_NAME
)

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=4096, num_classes=2, num_dense_layers=4,
    for_classification=False
)[1]
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.3, 0.3, 0.3, -1])

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(1.)
SCALAR_LOSS_STRING = (
    'custom_losses.scaled_mse_for_net_flux({0:.10f})'.format(1.)
)

OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([73, 14], dtype=int),
    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
        DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY:
        DENSE_LAYER_DROPOUT_RATES,
    u_net_pp_architecture.L2_WEIGHT_KEY: 1e-7
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING
}


def _run():
    """Makes U-net ++ template for prelim test.

    This is effectively the main method.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=OUTPUT_DIR_NAME
    )

    model_object = u_net_pp_architecture.create_model(
        option_dict=OPTION_DICT, vector_loss_function=VECTOR_LOSS_FUNCTION,
        num_output_channels=1, scalar_loss_function=SCALAR_LOSS_FUNCTION
    )

    model_file_name = '{0:s}/model_template.h5'.format(OUTPUT_DIR_NAME)
    print('Writing model to: "{0:s}"...'.format(model_file_name))
    model_object.save(
        filepath=model_file_name, overwrite=True, include_optimizer=True
    )

    loss_dict = {
        'conv_output': 'custom_losses.dual_weighted_mse()',
        'dense_output': SCALAR_LOSS_STRING
    }

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
        loss_function_or_dict=loss_dict,
        do_early_stopping=True, plateau_lr_multiplier=0.6
    )


if __name__ == '__main__':
    _run()
