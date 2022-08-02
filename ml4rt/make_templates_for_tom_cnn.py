"""Makes CNN templates for Tom Beucler."""

import os
import sys
import copy
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import cnn_architecture
import architecture_utils
import custom_losses
import file_system_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
OUTPUT_DIR_NAME = (
    '{0:s}/ml4rt_models/tom_experiment/cnn/templates'
).format(HOME_DIR_NAME)

VECTOR_LOSS_FUNCTION = custom_losses.dual_weighted_mse()
SCALAR_LOSS_FUNCTION = custom_losses.scaled_mse_for_net_flux(0.64)
LOSS_DICT = {
    'conv_output': 'custom_losses.dual_weighted_mse()',
    'dense_output': 'custom_losses.scaled_mse_for_net_flux(0.64)'
}

MODEL_DEPTHS = numpy.array([1, 2, 3, 4, 5, 6], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([2, 4, 8, 16, 32, 64], dtype=int)

DEFAULT_OPTION_DICT = {
    cnn_architecture.NUM_HEIGHTS_KEY: 127,
    cnn_architecture.NUM_INPUT_CHANNELS_KEY: 26,
    cnn_architecture.INNER_ACTIV_FUNCTION_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.INNER_ACTIV_FUNCTION_ALPHA_KEY: 0.2,
    cnn_architecture.CONV_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.CONV_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    cnn_architecture.DENSE_OUTPUT_ACTIV_FUNC_KEY:
        architecture_utils.RELU_FUNCTION_STRING,
    cnn_architecture.DENSE_OUTPUT_ACTIV_FUNC_ALPHA_KEY: 0.,
    cnn_architecture.L1_WEIGHT_KEY: 0.,
    cnn_architecture.L2_WEIGHT_KEY: 1e-7,
    cnn_architecture.USE_BATCH_NORM_KEY: True,
    # cnn_architecture.DENSE_LAYER_NEURON_NUMS_KEY: DENSE_LAYER_NEURON_COUNTS,
    cnn_architecture.DENSE_LAYER_DROPOUT_RATES_KEY: numpy.full(4, 0.)
}

DUMMY_GENERATOR_OPTION_DICT = {
    neural_net.VECTOR_TARGET_NORM_TYPE_KEY: None,
    neural_net.SCALAR_TARGET_NORM_TYPE_KEY: None
}


def _run():
    """Makes CNN templates for Tom Beucler.

    This is effectively the main method.
    """

    num_model_depths = len(MODEL_DEPTHS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    for i in range(num_model_depths):
        for j in range(num_channel_counts):
            option_dict = copy.deepcopy(DEFAULT_OPTION_DICT)

            these_coeffs = numpy.logspace(
                0, MODEL_DEPTHS[i] - 1, num=MODEL_DEPTHS[i], base=2.
            )
            channel_counts = numpy.minimum(
                these_coeffs * FIRST_LAYER_CHANNEL_COUNTS[j],
                64
            )
            option_dict[cnn_architecture.CONV_LAYER_CHANNEL_NUMS_KEY] = (
                numpy.round(channel_counts).astype(int)
            )

            option_dict[cnn_architecture.CONV_LAYER_DROPOUT_RATES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 0.)
            )
            option_dict[cnn_architecture.CONV_LAYER_FILTER_SIZES_KEY] = (
                numpy.full(MODEL_DEPTHS[i], 3, dtype=int)
            )

            num_flattened_features = (
                127 *
                option_dict[cnn_architecture.CONV_LAYER_CHANNEL_NUMS_KEY][-1]
            )

            option_dict[cnn_architecture.DENSE_LAYER_NEURON_NUMS_KEY] = (
                architecture_utils.get_dense_layer_dimensions(
                    num_input_units=num_flattened_features,
                    num_classes=2, num_dense_layers=4, for_classification=False
                )[1]
            )

            model_object = cnn_architecture.create_model(
                option_dict=option_dict,
                vector_loss_function=VECTOR_LOSS_FUNCTION,
                scalar_loss_function=SCALAR_LOSS_FUNCTION
            )

            model_file_name = (
                '{0:s}/num-levels={1:d}_num-first-layer-channels={2:02d}/'
                'model.h5'
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

            print('Writing metadata to: "{0:s}"...'.format(
                metafile_name
            ))
            neural_net._write_metafile(
                dill_file_name=metafile_name, num_epochs=100,
                num_training_batches_per_epoch=100,
                training_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                num_validation_batches_per_epoch=100,
                validation_option_dict=DUMMY_GENERATOR_OPTION_DICT,
                net_type_string=neural_net.CNN_TYPE_STRING,
                loss_function_or_dict=LOSS_DICT,
                do_early_stopping=True, plateau_lr_multiplier=0.6
            )


if __name__ == '__main__':
    _run()
