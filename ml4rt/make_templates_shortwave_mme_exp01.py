"""Makes templates for first shortwave experiment with MME.

MME = multi-model ensemble
"""

import os
import sys
import numpy
from keras import backend as K

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
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/shortwave_mme_experiment01/templates'
).format(HOME_DIR_NAME)

OUTER_ENSEMBLE_SIZE = 100
INNER_ENSEMBLE_SIZE = 25
MODEL_DEPTH = 3

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_crps()
SCALAR_LOSS_FUNCTION = custom_losses.unscaled_crps_for_net_flux()
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_crps()',
    'dense_output': 'custom_losses.unscaled_crps_for_net_flux()'
}

NUM_HEIGHTS_AT_DEEPEST_LAYER = 15
NUM_CHANNELS_AT_DEEPEST_LAYER = 1024

DENSE_LAYER_NEURON_COUNTS = architecture_utils.get_dense_layer_dimensions(
    num_input_units=(
        NUM_HEIGHTS_AT_DEEPEST_LAYER * NUM_CHANNELS_AT_DEEPEST_LAYER
    ),
    num_classes=2, num_dense_layers=4, for_classification=False
)[1]

DENSE_LAYER_NEURON_COUNTS[-1] = 2 * INNER_ENSEMBLE_SIZE
DENSE_LAYER_NEURON_COUNTS[-2] = max([
    DENSE_LAYER_NEURON_COUNTS[-2], DENSE_LAYER_NEURON_COUNTS[-1]
])

OPTION_DICT = {
    u_net_pp_architecture.INPUT_DIMENSIONS_KEY:
        numpy.array([127, 26], dtype=int),
    u_net_pp_architecture.NUM_LEVELS_KEY: MODEL_DEPTH,
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
    u_net_pp_architecture.USE_BATCH_NORM_KEY: True,
    u_net_pp_architecture.DENSE_LAYER_NEURON_NUMS_KEY:
        DENSE_LAYER_NEURON_COUNTS,
    u_net_pp_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.),
    u_net_pp_architecture.DENSE_LAYER_MC_DROPOUT_FLAGS_KEY:
        numpy.full(4, False, dtype=bool)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes templates for first shortwave experiment with MME.

    This is effectively the main method.
    """
    
    for i in range(OUTER_ENSEMBLE_SIZE):
        this_model_object = u_net_pp_architecture.create_model(
            option_dict=OPTION_DICT,
            vector_loss_function=VECTOR_LOSS_FUNCTION,
            use_deep_supervision=False, num_output_channels=1,
            scalar_loss_function=SCALAR_LOSS_FUNCTION,
            ensemble_size=INNER_ENSEMBLE_SIZE
        )

        print(K.eval(
            this_model_object.get_layer(name='block0-3_skip').weights[0]
        ))

        this_model_file_name = '{0:s}/ensemble-member={1:03d}/model.h5'.format(
            OUTPUT_DIR_NAME, i + 1
        )
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