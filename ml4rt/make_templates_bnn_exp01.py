"""Makes templates for first shortwave experiment with BNNs for UQ."""

import os
import sys
import copy
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
    '{0:s}/ml4rt_models/shortwave_bnn_experiment01/templates'
).format(HOME_DIR_NAME)

MODEL_DEPTH = 3
NUM_HEIGHTS_AT_DEEPEST_LAYER = 15
NUM_DENSE_LAYERS = 4

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(1.)
VECTOR_LOSS_FUNCTION_STRING = 'custom_losses.dual_weighted_mse()'
SCALAR_LOSS_FUNCTION_STRING = 'custom_losses.scaled_mse_for_net_flux(1.)'

LOSS_DICT = {
    'conv_output': VECTOR_LOSS_FUNCTION_STRING,
    'dense_output': SCALAR_LOSS_FUNCTION_STRING
}

POINT_ESTIMATE_TYPE_STRING = u_net_pp_architecture.POINT_ESTIMATE_TYPE_STRING
FLIPOUT_TYPE_STRING = u_net_pp_architecture.FLIPOUT_TYPE_STRING
REPARAMETERIZATION_TYPE_STRING = (
    u_net_pp_architecture.REPARAMETERIZATION_TYPE_STRING
)

FIRST_LAYER_CHANNEL_COUNTS = numpy.array([64, 128], dtype=int)
BAYESIAN_SKIP_LAYER_COUNTS = numpy.array([1, 2], dtype=int)
BAYESIAN_UPCONV_LAYER_COUNTS = numpy.array([1, 2], dtype=int)
BAYESIAN_DENSE_LAYER_COUNTS = numpy.array([1, 2, 3], dtype=int)
BAYESIAN_LAYER_TYPE_STRINGS = [
    FLIPOUT_TYPE_STRING, REPARAMETERIZATION_TYPE_STRING
]

NUM_TRAINING_EXAMPLES = int(5e5)

DEFAULT_OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: MODEL_DEPTH,
    u_net_pp_architecture.KL_SCALING_FACTOR_KEY: NUM_TRAINING_EXAMPLES ** -1,
    u_net_pp_architecture.CONV_LAYER_COUNTS_KEY:
        numpy.full(MODEL_DEPTH + 1, 1, dtype=int),
    u_net_pp_architecture.ENCODER_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH + 1, 0.),
    u_net_pp_architecture.ENCODER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH + 1, False, dtype=bool),
    u_net_pp_architecture.UPCONV_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_architecture.UPCONV_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, False, dtype=bool),
    u_net_pp_architecture.SKIP_DROPOUT_RATES_KEY:
        numpy.full(MODEL_DEPTH, 0.),
    u_net_pp_architecture.SKIP_MC_DROPOUT_FLAGS_KEY:
        numpy.full(MODEL_DEPTH, False, dtype=bool),
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

    # TODO(thunderhoser): Might want to experiment with no-batch-norm in future.
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(4, False, dtype=bool),
    u_net_pp_architecture.NUM_OUTPUT_CHANNELS_KEY: 1,
    u_net_pp_architecture.VECTOR_LOSS_FUNCTION_KEY: VECTOR_LOSS_FUNCTION,
    u_net_pp_architecture.SCALAR_LOSS_FUNCTION_KEY: SCALAR_LOSS_FUNCTION
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes templates for first shortwave experiment with BNNs for UQ.

    This is effectively the main method.
    """

    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)
    num_skip_layer_counts = len(BAYESIAN_SKIP_LAYER_COUNTS)
    num_upconv_layer_counts = len(BAYESIAN_UPCONV_LAYER_COUNTS)
    num_dense_layer_counts = len(BAYESIAN_DENSE_LAYER_COUNTS)
    num_layer_types = len(BAYESIAN_LAYER_TYPE_STRINGS)

    for i in range(num_channel_counts):
        for j in range(num_skip_layer_counts):
            for k in range(num_upconv_layer_counts):
                for l in range(num_dense_layer_counts):
                    for m in range(num_layer_types):
                        option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

                        this_log = int(numpy.round(
                            numpy.log2(FIRST_LAYER_CHANNEL_COUNTS[i])
                        ))
                        these_channel_counts = numpy.logspace(
                            this_log, this_log + MODEL_DEPTH,
                            num=MODEL_DEPTH + 1, dtype=int
                        )

                        print(these_channel_counts)

                        upconv_layer_type_strings = (
                            [POINT_ESTIMATE_TYPE_STRING] * MODEL_DEPTH
                            + [BAYESIAN_LAYER_TYPE_STRINGS[m]] *
                            BAYESIAN_UPCONV_LAYER_COUNTS[k]
                        )
                        upconv_layer_type_strings = (
                            upconv_layer_type_strings[-MODEL_DEPTH:]
                        )

                        skip_layer_type_strings = (
                            [POINT_ESTIMATE_TYPE_STRING] * MODEL_DEPTH
                            + [BAYESIAN_LAYER_TYPE_STRINGS[m]] *
                            BAYESIAN_SKIP_LAYER_COUNTS[j]
                        )
                        skip_layer_type_strings = (
                            skip_layer_type_strings[-MODEL_DEPTH:]
                        )

                        dense_neuron_counts = (
                            architecture_utils.get_dense_layer_dimensions(
                                num_input_units=(
                                    NUM_HEIGHTS_AT_DEEPEST_LAYER *
                                    these_channel_counts[-1]
                                ),
                                num_classes=2,
                                num_dense_layers=4,
                                for_classification=False
                            )[1]
                        )

                        dense_layer_type_strings = (
                            [POINT_ESTIMATE_TYPE_STRING] * NUM_DENSE_LAYERS
                            + [BAYESIAN_LAYER_TYPE_STRINGS[m]] *
                            BAYESIAN_DENSE_LAYER_COUNTS[l]
                        )
                        dense_layer_type_strings = (
                            dense_layer_type_strings[-NUM_DENSE_LAYERS:]
                        )

                        option_dict.update({
                            u_net_pp_architecture.CHANNEL_COUNTS_KEY:
                                these_channel_counts,
                            u_net_pp_architecture.UPCONV_BNN_LAYER_TYPES_KEY:
                                upconv_layer_type_strings,
                            u_net_pp_architecture.SKIP_BNN_LAYER_TYPES_KEY:
                                skip_layer_type_strings,
                            u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
                                dense_neuron_counts,
                            u_net_pp_architecture.DENSE_BNN_LAYER_TYPES_KEY:
                                dense_layer_type_strings,
                            u_net_pp_architecture.CONV_OUTPUT_BNN_LAYER_TYPE_KEY:
                                BAYESIAN_LAYER_TYPE_STRINGS[m]
                        })

                        model_object = (
                            u_net_pp_architecture.create_bayesian_model(
                                option_dict
                            )
                        )

                        model_file_name = (
                            '{0:s}/num-first-layer-channels={1:03d}_'
                            'num-bayesian-skip-layers={2:d}_'
                            'num-bayesian-upconv-layers={3:d}_'
                            'num-bayesian-dense-layers={4:d}_'
                            'bayesian-layer-type={5:s}/model.h5'
                        ).format(
                            OUTPUT_DIR_NAME,
                            FIRST_LAYER_CHANNEL_COUNTS[i],
                            BAYESIAN_SKIP_LAYER_COUNTS[j],
                            BAYESIAN_UPCONV_LAYER_COUNTS[k],
                            BAYESIAN_DENSE_LAYER_COUNTS[l],
                            BAYESIAN_LAYER_TYPE_STRINGS[m]
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

                        option_dict[
                            u_net_pp_architecture.VECTOR_LOSS_FUNCTION_KEY
                        ] = VECTOR_LOSS_FUNCTION_STRING

                        option_dict[
                            u_net_pp_architecture.SCALAR_LOSS_FUNCTION_KEY
                        ] = SCALAR_LOSS_FUNCTION_STRING

                        print('Writing metadata to: "{0:s}"...'.format(
                            metafile_name
                        ))
                        neural_net._write_metafile(
                            dill_file_name=metafile_name, num_epochs=100,
                            num_training_batches_per_epoch=100,
                            training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                            num_validation_batches_per_epoch=100,
                            validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                            net_type_string=neural_net.U_NET_TYPE_STRING,
                            loss_function_or_dict=LOSS_DICT,
                            do_early_stopping=True, plateau_lr_multiplier=0.6,
                            bnn_architecture_dict=option_dict
                        )


if __name__ == '__main__':
    _run()
