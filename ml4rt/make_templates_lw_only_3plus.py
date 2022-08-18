"""Makes U-net 3+ templates for experiment with longwave only."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_ppp_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/lw_only_experiment_3plus/templates'
).format(HOME_DIR_NAME)

NUM_LEVELS = 4
VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=7 * 1024, num_classes=2, num_dense_layers=4,
    for_classification=False
)[1]

FLUX_LOSS_SCALING_FACTORS = numpy.linspace(0.02, 1, num=50, dtype=float)

DEFAULT_OPTION_DICT = {
    u_net_ppp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 23], dtype=int),
    u_net_ppp_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_ppp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    u_net_ppp_architecture.CHANNEL_COUNTS_KEY: numpy.round(
        numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    ).astype(int),
    u_net_ppp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.full(NUM_LEVELS + 1, 0.),
    u_net_ppp_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_ppp_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_ppp_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_ppp_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_ppp_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_ppp_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_ppp_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY: None,
    u_net_ppp_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_ppp_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_ppp_architecture.L1_WEIGHT_KEY: 0.,
    u_net_ppp_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_ppp_architecture.USE_BATCH_NORM_KEY: True,
    u_net_ppp_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_ppp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes U-net 3+ templates for experiment with longwave only.

    This is effectively the main method.
    """

    num_scaling_factors = len(FLUX_LOSS_SCALING_FACTORS)

    for i in range(num_scaling_factors):
        this_scalar_loss_function = custom_losses.scaled_mse_for_net_fluxes(
            scaling_factor=FLUX_LOSS_SCALING_FACTORS[i],
            down_flux_indices=numpy.array([0], dtype=int),
            up_flux_indices=numpy.array([1], dtype=int)
        )
        this_scalar_loss_string = (
            'custom_losses.scaled_mse_for_net_fluxes(scaling_factor={0:.10f}, '
            'down_flux_indices=numpy.array([0], dtype=int), '
            'up_flux_indices=numpy.array([1], dtype=int))'.format(
                FLUX_LOSS_SCALING_FACTORS[i]
            )
        )
        this_loss_dict = {
            'conv_output': 'custom_losses.dual_weighted_mse()',
            'dense_output': this_scalar_loss_string
        }

        this_model_object = u_net_ppp_architecture.create_model(
            option_dict=DEFAULT_OPTION_DICT,
            vector_loss_function=VECTOR_LOSS_FUNCTION,
            use_deep_supervision=False, num_output_channels=1,
            scalar_loss_function=this_scalar_loss_function
        )
        this_model_file_name = (
            '{0:s}/flux-loss-scaling-factor={1:.2f}/model.h5'
        ).format(OUTPUT_DIR_NAME, FLUX_LOSS_SCALING_FACTORS[i])

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=this_model_file_name
        )

        print('Writing model to: "{0:s}"...'.format(
            this_model_file_name
        ))
        this_model_object.save(
            filepath=this_model_file_name, overwrite=True,
            include_optimizer=True
        )

        this_metafile_name = neural_net.find_metafile(
            model_dir_name=os.path.split(this_model_file_name)[0],
            raise_error_if_missing=False
        )

        print('Writing metadata to: "{0:s}"...'.format(
            this_metafile_name
        ))
        neural_net._write_metafile(
            dill_file_name=this_metafile_name, num_epochs=100,
            num_training_batches_per_epoch=100,
            training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
            num_validation_batches_per_epoch=100,
            validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
            net_type_string=neural_net.U_NET_TYPE_STRING,
            loss_function_or_dict=this_loss_dict,
            do_early_stopping=True, plateau_lr_multiplier=0.6
        )


if __name__ == '__main__':
    _run()
