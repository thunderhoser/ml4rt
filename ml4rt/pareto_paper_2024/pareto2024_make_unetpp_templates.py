"""Makes U-net++ templates for 2024 Pareto-front paper with Tom Beucler."""

import copy
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import u_net_pp_architecture
from ml4rt.machine_learning import keras_losses as custom_losses

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(
    scaling_factor=0.64
)

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    # u_net_pp_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    # u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
    #     numpy.full(NUM_LEVELS + 1, 2, dtype=int),
    # u_net_pp_architecture.CHANNEL_COUNTS_KEY: numpy.round(
    #     numpy.logspace(6, 10, num=NUM_LEVELS + 1, base=2.)
    # ).astype(int),
    # u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
    #     numpy.full(NUM_LEVELS + 1, 0.),
    # u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    # u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY: numpy.full(NUM_LEVELS, 0.),
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_pp_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
    u_net_pp_architecture.PENULTIMATE_MC_DROPOUT_FLAG_KEY: False,
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
    u_net_pp_architecture.USE_RESIDUAL_BLOCKS_KEY: False,
    # u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY: numpy.full(
        4, False, dtype=bool
    ),
    u_net_pp_architecture.NUM_OUTPUT_WAVELENGTHS_KEY: 1,
    u_net_pp_architecture.ENSEMBLE_SIZE_KEY: 1,
    u_net_pp_architecture.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    u_net_pp_architecture.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION,
    u_net_pp_architecture.USE_DEEP_SUPERVISION_KEY: False,
    u_net_pp_architecture.INCLUDE_MASK_KEY: False
}


def _run():
    """Makes U-net++ templates for 2024 Pareto-front paper with Tom Beucler.

    This is effectively the main method.
    """

    num_model_depths = len(MODEL_DEPTHS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    for i in range(num_model_depths):
        for j in range(num_channel_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            option_dict[u_net_pp_architecture.NUM_LEVELS_KEY] = MODEL_DEPTHS[i]
            option_dict[u_net_pp_architecture.CONV_LAYER_COUNTS_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 2, dtype=int)
            )

            these_coeffs = numpy.logspace(
                0, MODEL_DEPTHS[i], num=MODEL_DEPTHS[i] + 1, base=2.
            )
            these_coeffs = numpy.round(these_coeffs).astype(int)
            option_dict[u_net_pp_architecture.CHANNEL_COUNTS_KEY] = (
                these_coeffs * FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            option_dict[u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, 0.)
            )
            option_dict[u_net_pp_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY] = (
                numpy.full(MODEL_DEPTHS[i] + 1, False, dtype=bool)
            )
            option_dict[u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 0.)
            )
            option_dict[u_net_pp_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY] = (
                numpy.full(MODEL_DEPTHS[i], False, dtype=bool)
            )
            option_dict[u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 0.)
            )
            option_dict[u_net_pp_architecture.SKIP_MC_DROPOUT_FLAGS_KEY] = (
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
                option_dict[u_net_pp_architecture.CHANNEL_COUNTS_KEY][-1]
            )

            option_dict[u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY] = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=num_flattened_features,
                    num_classes=2, num_dense_layers=4, for_classification=False
                )[1]
            )

            model_object = u_net_pp_architecture.create_model(option_dict)

            model_file_name = (
                'num-levels={0:d}_num-first-layer-channels={1:02d}/model.keras'
            ).format(
                MODEL_DEPTHS[i],
                FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            file_system_utils.mkdir_recursive_if_necessary(
                file_name=model_file_name
            )
            print('Writing model to: "{0:s}"...'.format(model_file_name))
            model_object.save(
                filepath=model_file_name, overwrite=True, include_optimizer=True
            )


if __name__ == '__main__':
    _run()
