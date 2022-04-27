"""Makes architecture for best U-net++ from Experiment 2 in journal paper."""

import copy
import numpy
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.machine_learning import u_net_pp_architecture
from ml4rt.machine_learning import keras_losses as custom_losses

DENSE_LAYER_COUNT = 4
DENSE_LAYER_DROPOUT_RATE = 0.
L2_WEIGHT = 10 ** -7

# Number of levels in U-net++.
NUM_LEVELS = 4

# Loss function for heating-rate profiles.
VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()

# Loss function for flux components.
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(scaling_factor=1.)

# The "penultimate" layer is the second-last convolutional layer, which reduces
# the number of channels to 2.  DEFAULT_OPTION_DICT contains fixed
# hyperparameters, which were not varied during the hyperparameter experiment
# presented in the paper.
DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([73, 14], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: NUM_LEVELS,
    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.array([2, 2, 2, 2, 2], dtype=int),
    u_net_pp_architecture.CHANNEL_COUNTS_KEY:
        numpy.array([64, 128, 256, 512, 1024], dtype=int),
    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.array([0, 0, 0, 0, 0], dtype=float),
    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
        numpy.array([0, 0, 0, 0], dtype=float),
    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
        numpy.array([0, 0, 0, 0], dtype=float),
    u_net_pp_architecture.INCLUDE_PENULTIMATE_KEY: True,
    u_net_pp_architecture.PENULTIMATE_DROPOUT_RATE_KEY: 0.,
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
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True
}


def _run():
    """Makes architecture for best U-net++ from Experiment 2 in journal paper.

    This is effectively the main method.
    """

    dense_layer_neuron_counts = architecture_utils.get_dense_layer_dimensions(
        num_input_units=4096, num_classes=2,
        num_dense_layers=DENSE_LAYER_COUNT, for_classification=False
    )[1]

    dense_layer_dropout_rates = numpy.full(
        DENSE_LAYER_COUNT, DENSE_LAYER_DROPOUT_RATE
    )

    # Never use dropout for the last dense layer.
    dense_layer_dropout_rates[-1] = numpy.nan

    option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)
    option_dict[u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY] = (
        dense_layer_neuron_counts
    )
    option_dict[u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY] = (
        dense_layer_dropout_rates
    )
    option_dict[u_net_pp_architecture.L2_WEIGHT_KEY] = L2_WEIGHT

    # This next command creates the U-net++ architecture and compiles the
    # U-net++.  Thus, it returns a U-net++ that is completely ready to train.
    # It also prints a very long flow chart, containing details on each layer in
    # the model.  I have not included this flow chart in the paper, because it
    # is difficult to read and I feel that Figures 1-2 do a better job of
    # documenting the architecture.

    model_object = u_net_pp_architecture.create_model(
        option_dict=option_dict,
        vector_loss_function=VECTOR_LOSS_FUNCTION,
        num_output_channels=1,
        scalar_loss_function=SCALAR_LOSS_FUNCTION
    )


if __name__ == '__main__':
    _run()
