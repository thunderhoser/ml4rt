"""Converts neural net and isotonic regression to a single Tensorflow model.

To convert to ONNX after this, do the following from the command line:

python3 -m tf2onnx.convert --saved-model ${output_nn_dir_name} \
--output ${onnx_file_name}
"""

import os
import argparse
import keras
import tensorflow
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import isotonic_regression

INPUT_NN_FILE_ARG_NAME = 'input_nn_file_name'
ISO_REG_FILE_ARG_NAME = 'input_iso_reg_file_name'
FLATTEN_OUTPUT_ARG_NAME = 'flatten_output'
OUTPUT_NN_DIR_ARG_NAME = 'output_nn_dir_name'

INPUT_NN_FILE_HELP_STRING = (
    'Path to basic trained neural-net model, without isotonic regression.  '
    'Will be read by `neural_net.read_model`.'
)
ISO_REG_FILE_HELP_STRING = (
    'Path to trained suite of isotonic-regression models.  Will be read by '
    '`isotonic_regression.read_file`.'
)
FLATTEN_OUTPUT_HELP_STRING = (
    'Boolean flag.  If 1, will flatten output layer to be two-dimensional '
    '(batch_sample, atomic_variable).'
)
OUTPUT_NN_DIR_HELP_STRING = (
    'Path to output neural-net model, containing isotonic regression.  Will be '
    'written to this directory by `tensorflow.saved_model.save`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_NN_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_NN_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISO_REG_FILE_ARG_NAME, type=str, required=True,
    help=ISO_REG_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FLATTEN_OUTPUT_ARG_NAME, type=int, required=False, default=0,
    help=FLATTEN_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_NN_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_NN_DIR_HELP_STRING
)


def _run(input_nn_file_name, iso_reg_file_name, flatten_output,
         output_nn_dir_name):
    """Converts neural net and isotonic regression to a single Tensorflow model.

    This is effectively the main method.

    :param input_nn_file_name: See documentation at top of this script.
    :param iso_reg_file_name: Same.
    :param flatten_output: Same.
    :param output_nn_dir_name: Same.
    """

    print('Reading basic neural net from: "{0:s}"...'.format(
        input_nn_file_name
    ))
    nn_model_object = neural_net.read_model(input_nn_file_name)

    nn_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(input_nn_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading suite of isotonic-regression models from: "{0:s}"...'.format(
        iso_reg_file_name
    ))
    scalar_ir_model_matrix, vector_ir_model_matrix = (
        isotonic_regression.read_file(iso_reg_file_name)
    )

    nn_model_object = isotonic_regression.add_ir_to_neural_net(
        nn_model_object=nn_model_object,
        nn_metafile_name=nn_metafile_name,
        scalar_model_object_matrix=scalar_ir_model_matrix,
        vector_model_object_matrix=vector_ir_model_matrix
    )

    hr_output_layer_object = nn_model_object.output[0][..., 0]
    flux_output_layer_object = nn_model_object.output[1]
    flux_output_layer_object = keras.layers.Permute(dims=(2, 1))(
        flux_output_layer_object
    )
    output_layer_object = keras.layers.Concatenate(axis=1)(
        [hr_output_layer_object, flux_output_layer_object]
    )

    if flatten_output:
        output_layer_object = keras.layers.Reshape(target_shape=(-1,))(
            output_layer_object
        )

    nn_model_object = keras.models.Model(
        inputs=nn_model_object.input, outputs=output_layer_object
    )
    nn_model_object.summary()

    print('Writing NN/IR model to: "{0:s}"...'.format(output_nn_dir_name))
    tensorflow.saved_model.save(nn_model_object, output_nn_dir_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_nn_file_name=getattr(INPUT_ARG_OBJECT, INPUT_NN_FILE_ARG_NAME),
        iso_reg_file_name=getattr(INPUT_ARG_OBJECT, ISO_REG_FILE_ARG_NAME),
        flatten_output=bool(getattr(INPUT_ARG_OBJECT, FLATTEN_OUTPUT_ARG_NAME)),
        output_nn_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_NN_DIR_ARG_NAME)
    )
