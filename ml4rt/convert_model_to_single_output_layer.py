"""Converts trained model to have a single output layer.

The single output layer will contain predictions of both heating rate and flux.
"""

import os
import sys
import argparse
import keras.layers
import keras.models
import tensorflow

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import neural_net

INPUT_FILE_ARG_NAME = 'input_keras_file_name'
OUTPUT_DIR_ARG_NAME = 'output_tensorflow_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input model (in .keras format, readable by '
    '`neural_net.read_model`).  This model should contain two output layers: '
    'one for heating rates and one for fluxes.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  The single-output-layer model will be saved '
    'here, using `tensorflow.saved_model.save`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Converts trained model to have a single output layer.

    This is effectively the main method.

    :param input_file_name: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    print('Reading trained Keras model from: "{0:s}"...'.format(
        input_file_name
    ))
    model_object = neural_net.read_model(input_file_name)
    model_object.summary()

    input_layer_objects = model_object.input
    conv_output_layer_object = model_object.get_layer('conv_output').output
    dense_output_layer_object = model_object.get_layer('dense_output').output

    # TODO(thunderhoser): Asserts that NN is deterministic, i.e., does not
    # produce an ensemble.  I could make this work for ensemble NNs, though.
    assert len(conv_output_layer_object.shape) == 4
    assert len(dense_output_layer_object.shape) == 3

    # Squeeze out the channel dimension (only one channel -- heating rate).
    conv_output_layer_object = keras.layers.Reshape(
        target_shape=conv_output_layer_object.shape[1:-1]
    )(conv_output_layer_object)

    # Permute height and wavelength dimensions, so that conv layer has
    # dimensions E x H x W and dense layer has dimensions E x F x W -- where
    # E = number of examples, H = number of heights,
    # F = number of flux variables, and W = number of wavelengths.
    dense_output_layer_object = keras.layers.Permute((2, 1))(
        dense_output_layer_object
    )

    # Concatenate the two output layers.  Now we have one output layer.
    single_output_layer_object = keras.layers.Concatenate(axis=1)(
        [conv_output_layer_object, dense_output_layer_object]
    )

    # Create the new model and save it.
    model_object = keras.models.Model(
        inputs=input_layer_objects, outputs=single_output_layer_object
    )
    model_object.summary()

    parent_dir_name = '/'.join(output_dir_name.split('/')[:-1])
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=parent_dir_name
    )

    print('Saving trained TensorFlow model to: "{0:s}"...'.format(
        output_dir_name
    ))
    tensorflow.saved_model.save(model_object, output_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
