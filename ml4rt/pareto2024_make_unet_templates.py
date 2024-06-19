"""Makes U-net templates for 2024 Pareto-front paper with Tom Beucler."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import u_net_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/pareto2024_experiment/u_net/templates'
).format(HOME_DIR_NAME)

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(0.64)
VECTOR_LOSS_FUNCTION_STRING = 'custom_losses.dual_weighted_mse()'
SCALAR_LOSS_FUNCTION_STRING = 'custom_losses.scaled_mse_for_net_flux(0.64)'

LOSS_DICT = {
    'conv_output': VECTOR_LOSS_FUNCTION_STRING,
    'dense_output': SCALAR_LOSS_FUNCTION_STRING
}

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

DEFAULT_OPTION_DICT = {
    u_net_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    # u_net_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    # u_net_architecture.CONV_LAYER_COUNTS_KEY:
    #     numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    # u_net_architecture.CHANNEL_COUNTS_KEY: numpy.round(
    #     numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    # ).astype(int),
    # u_net_architecture.ENCODER_DROPOUT_RATES_KEY:
    #     numpy.full(NUM_LEVELS + 1, 0.),
    # u_net_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    # u_net_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_architecture.PENULTIMATE_MC_DROPOUT_FLAG_KEY: False,
    u_net_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    u_net_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    u_net_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    u_net_architecture.L1_WEIGHT_KEY: 0.,
    u_net_architecture.L2_WEIGHT_KEY: 1e-7,
    u_net_architecture.USE_BATCH_NORM_KEY: True,
    u_net_architecture.USE_RESIDUAL_BLOCKS_KEY: False,
    # u_net_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        4, False, dtype=bool
    ),
    u_net_architecture.NUM_OUTPUT_WAVELENGTHS_KEY: 1,
    u_net_architecture.ENSEMBLE_SIZE_KEY: 1,
    u_net_architecture.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    u_net_architecture.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.NORMALIZE_PREDICTORS_KEY: True,
    neural_net.NORMALIZE_VECTOR_TARGETS_KEY: False,
    neural_net.NORMALIZE_SCALAR_TARGETS_KEY: False
}


def _run():
    """Makes U-net templates for 2024 Pareto-front paper with Tom Beucler.

    This is effectively the main method.
    """

    num_model_depths = len(MODEL_DEPTHS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    for i in range(num_model_depths):
        for j in range(num_channel_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            option_dict[u_net_architecture.NUM_LEVELS_KEY] = MODEL_DEPTHS[i]
            option_dict[u_net_architecture.CONV_LAYER_COUNTS_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 2, dtype=int)
            )

            these_coeffs = numpy.logspace(
                0, MODEL_DEPTHS[i], num=MODEL_DEPTHS[i] + 1, base=2.
            )
            these_coeffs = numpy.round(these_coeffs).astype(int)
            option_dict[u_net_architecture.CHANNEL_COUNTS_KEY] = (
                these_coeffs * FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            option_dict[u_net_architecture.ENCODER_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 0.)
            )
            option_dict[u_net_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, False, dtype=bool)
            )
            option_dict[u_net_architecture.UPCONV_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 0.)
            )
            option_dict[u_net_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY] = (
                numpy.full(MODEL_DEPTHS[i], False, dtype=bool)
            )
            option_dict[u_net_architecture.SKIP_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 0.)
            )
            option_dict[u_net_architecture.SKIP_MC_DROPOUT_FLAGS_KEY] = (
                numpy.full(MODEL_DEPTHS[i], False, dtype=bool)
            )

            if MODEL_DEPTHS[i] == 5:
                final_num_heights = 3
            elif MODEL_DEPTHS[i] == 4:
                final_num_heights = 7
            else:
                final_num_heights = 15

            num_flattened_features = (
                final_num_heights *
                option_dict[u_net_architecture.CHANNEL_COUNTS_KEY][-1]
            )

            option_dict[u_net_architecture.DENSE_LAYER_NEURON_NUMS_KEY] = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=num_flattened_features,
                    num_classes=2, num_dense_layers=4, for_classification=False
                )[1]
            )

            model_object = u_net_architecture.create_model(option_dict)

            model_file_name = (
                '{0:s}/num-levels={1:d}_num-first-layer-channels={2:02d}/'
                'model.keras'
            ).format(
                OUTPUT_DIR_NAME, MODEL_DEPTHS[i], FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )

            print('Writing model to: "{0:s}"...'.format(
                model_file_name
            ))
            model_object.save(
                filepath=model_file_name, overwrite=True,
                include_optimizer=True
            )

            metafile_name = neural_net.find_metafile(
                model_dir_name=os.path.split(model_file_name)[0],
                raise_error_if_missing=False
            )

            option_dict[u_net_architecture.VECTOR_LOSS_FUNCTION_KEY] = (
                VECTOR_LOSS_FUNCTION_STRING
            )
            option_dict[u_net_architecture.SCALAR_LOSS_FUNCTION_KEY] = (
                SCALAR_LOSS_FUNCTION_STRING
            )

            print('Writing metadata to: "{0:s}"...'.format(
                metafile_name
            ))
            neural_net._write_metafile(
                dill_file_name=metafile_name,
                num_epochs=100,
                num_training_batches_per_epoch=100,
                training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                num_validation_batches_per_epoch=100,
                validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                loss_function_or_dict=LOSS_DICT,
                early_stopping_patience_epochs=200,
                plateau_lr_multiplier=0.6,
                dense_architecture_dict=None,
                cnn_architecture_dict=None,
                bnn_architecture_dict=None,
                u_net_architecture_dict=option_dict,
                u_net_plusplus_architecture_dict=None,
                u_net_3plus_architecture_dict=None,
                use_ryan_architecture=False
            )


if __name__ == '__main__':
    _run()
