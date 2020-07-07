"""Helper methods for Grad-CAM (gradient-weighted class-activation maps)."""

import numpy
import netCDF4
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import gradcam as gradcam_utils

DEFAULT_IDEAL_ACTIVATION = 2.

MODEL_FILE_KEY = 'model_file_name'
ACTIVATION_LAYER_KEY = 'activation_layer_name'
VECTOR_OUT_LAYER_KEY = 'vector_output_layer_name'
OUTPUT_NEURONS_KEY = 'output_neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
TARGET_DIMENSION_KEY = 'target'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

EXAMPLE_IDS_KEY = 'example_id_strings'
CLASS_ACTIVATIONS_KEY = 'class_activation_matrix'


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def check_metadata(
        activation_layer_name, vector_output_layer_name, output_neuron_indices,
        ideal_activation):
    """Checks metadata for errors.

    :param activation_layer_name: Name of activation layer.
    :param vector_output_layer_name: Name of layer that outputs predictions for
        vector target variables.
    :param output_neuron_indices: length-2 numpy array with indices of output
        neuron (height index, channel index).  Class activation will be computed
        with respect to the output of this neuron.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (output_neuron_activation - ideal_activation)**2.
    """

    error_checking.assert_is_string(activation_layer_name)
    error_checking.assert_is_string(vector_output_layer_name)

    error_checking.assert_is_integer_numpy_array(output_neuron_indices)
    error_checking.assert_is_geq_numpy_array(output_neuron_indices, 0)
    error_checking.assert_is_numpy_array(
        output_neuron_indices, exact_dimensions=numpy.array([2], dtype=int)
    )

    error_checking.assert_is_not_nan(ideal_activation)


def run_gradcam(
        model_object, predictor_matrix, activation_layer_name,
        vector_output_layer_name, output_neuron_indices,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Runs the Grad-CAM algorithm.

    H = number of heights
    P = number of predictor variables

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: H-by-P numpy array of predictor values.
    :param activation_layer_name: See doc for `check_metadata`.
    :param vector_output_layer_name: Same.
    :param output_neuron_indices: Same.
    :param ideal_activation: same.
    :return: class_activations: length-H numpy array of class activations.
    """

    # TODO(thunderhoser): Eventually make this work for dense (scalar) output
    # layers as well.

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

    check_metadata(
        activation_layer_name=activation_layer_name,
        vector_output_layer_name=vector_output_layer_name,
        output_neuron_indices=output_neuron_indices,
        ideal_activation=ideal_activation
    )

    # Set up loss function.
    output_tensor = model_object.get_layer(
        name=vector_output_layer_name
    ).output[:, output_neuron_indices[0], output_neuron_indices[1]]

    # TODO(thunderhoser): Is this right?
    # loss_tensor = (output_tensor - ideal_activation) ** 2
    loss_tensor = output_tensor

    # Set up gradient function.
    layer_activation_tensor = (
        model_object.get_layer(name=activation_layer_name).output
    )
    gradient_tensor = (
        K.gradients(loss_tensor, [layer_activation_tensor])[0]
    )
    gradient_tensor = _normalize_tensor(gradient_tensor)
    gradient_function = K.function(
        [model_object.input], [layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    layer_activation_matrix, gradient_matrix = gradient_function(
        [predictor_matrix]
    )
    layer_activation_matrix = layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map in activation layer's space.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=0)
    class_activations = numpy.full(layer_activation_matrix.shape[0], 0.)
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activations += (
            mean_weight_by_filter[k] * layer_activation_matrix[:, k]
        )

    num_input_heights = predictor_matrix.shape[1]
    class_activation_matrix = gradcam_utils._upsample_cam(
        class_activation_matrix=class_activations,
        new_dimensions=numpy.array([num_input_heights], dtype=int)
    )

    return numpy.maximum(class_activation_matrix, 0.)


def write_standard_file(
        netcdf_file_name, class_activation_matrix, example_id_strings,
        model_file_name, activation_layer_name, vector_output_layer_name,
        output_neuron_indices, ideal_activation):
    """Writes standard (non-averaged) class-activation maps to NetCDF file.

    E = number of examples
    H = number of heights

    :param netcdf_file_name: Path to output file.
    :param class_activation_matrix: E-by-H numpy array of class activations.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param activation_layer_name: See doc for `check_metadata`.
    :param vector_output_layer_name: Same.
    :param output_neuron_indices: Same.
    :param ideal_activation: Same.
    """

    # Check input args.
    check_metadata(
        activation_layer_name=activation_layer_name,
        vector_output_layer_name=vector_output_layer_name,
        output_neuron_indices=output_neuron_indices,
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    error_checking.assert_is_geq_numpy_array(class_activation_matrix, 0.)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, num_dimensions=2
    )

    num_examples = len(example_id_strings)
    num_heights = class_activation_matrix.shape[1]
    expected_dim = numpy.array([num_examples, num_heights], dtype=int)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, exact_dimensions=expected_dim
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(ACTIVATION_LAYER_KEY, activation_layer_name)
    dataset_object.setncattr(VECTOR_OUT_LAYER_KEY, vector_output_layer_name)
    dataset_object.setncattr(OUTPUT_NEURONS_KEY, output_neuron_indices)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, num_heights)

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in example_id_strings
        ]))

    dataset_object.createDimension(EXAMPLE_ID_CHAR_DIM_KEY, num_id_characters)

    this_string_format = 'S{0:d}'.format(num_id_characters)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array
    )

    dataset_object.createVariable(
        CLASS_ACTIVATIONS_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY)
    )
    dataset_object.variables[CLASS_ACTIVATIONS_KEY][:] = (
        class_activation_matrix
    )

    dataset_object.close()


def read_standard_file(netcdf_file_name):
    """Reads standard (non-averaged) class-activation maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: gradcam_dict: Dictionary with the following keys.
    gradcam_dict['class_activation_matrix']: See doc for `write_standard_file`.
    gradcam_dict['example_id_strings']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['activation_layer_name']: Same.
    gradcam_dict['vector_output_layer_name']: Same.
    gradcam_dict['output_neuron_indices']: Same.
    gradcam_dict['ideal_activation']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    gradcam_dict = {
        CLASS_ACTIVATIONS_KEY:
            dataset_object.variables[CLASS_ACTIVATIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        ACTIVATION_LAYER_KEY:
            str(getattr(dataset_object, ACTIVATION_LAYER_KEY)),
        VECTOR_OUT_LAYER_KEY:
            str(getattr(dataset_object, VECTOR_OUT_LAYER_KEY)),
        OUTPUT_NEURONS_KEY: numpy.array(
            getattr(dataset_object, OUTPUT_NEURONS_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    dataset_object.close()
    return gradcam_dict


def write_all_targets_file(
        netcdf_file_name, class_activation_matrix, example_id_strings,
        model_file_name, activation_layer_name, vector_output_layer_name,
        ideal_activation):
    """Writes class-activation maps for all target variables to NetCDF file.

    E = number of examples
    H = number of heights
    T_v = number of vector targets

    :param netcdf_file_name: Path to output file.
    :param class_activation_matrix: numpy array (E x H x H x T_v) of class
        activations.  class_activation_matrix[:, :, j, k] is the
        class-activation map for the [k]th target variable at the [j]th height.
    :param example_id_strings: See doc for `write_standard_file`.
    :param model_file_name: Same.
    :param activation_layer_name: Same.
    :param vector_output_layer_name: Same.
    :param ideal_activation: Same.
    """

    # Check input args.
    check_metadata(
        activation_layer_name=activation_layer_name,
        vector_output_layer_name=vector_output_layer_name,
        output_neuron_indices=numpy.array([0, 0], dtype=int),
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_string(model_file_name)

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    error_checking.assert_is_geq_numpy_array(class_activation_matrix, 0.)
    error_checking.assert_is_numpy_array(
        class_activation_matrix, num_dimensions=4
    )

    num_examples = len(example_id_strings)
    expected_dim = numpy.array(
        (num_examples,) + class_activation_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        class_activation_matrix, exact_dimensions=expected_dim
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(ACTIVATION_LAYER_KEY, activation_layer_name)
    dataset_object.setncattr(VECTOR_OUT_LAYER_KEY, vector_output_layer_name)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)

    num_heights = class_activation_matrix.shape[1]
    num_targets = class_activation_matrix.shape[-1]
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, num_heights)
    dataset_object.createDimension(TARGET_DIMENSION_KEY, num_targets)

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in example_id_strings
        ]))

    dataset_object.createDimension(EXAMPLE_ID_CHAR_DIM_KEY, num_id_characters)

    this_string_format = 'S{0:d}'.format(num_id_characters)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array
    )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
        TARGET_DIMENSION_KEY
    )
    dataset_object.createVariable(
        CLASS_ACTIVATIONS_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[CLASS_ACTIVATIONS_KEY][:] = (
        class_activation_matrix
    )

    dataset_object.close()


def read_all_targets_file(netcdf_file_name):
    """Reads class-activation maps for all target variables from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: gradcam_dict: Dictionary with the following keys.
    gradcam_dict['class_activation_matrix']: See doc for
        `write_all_targets_file`.
    gradcam_dict['example_id_strings']: Same.
    gradcam_dict['model_file_name']: Same.
    gradcam_dict['activation_layer_name']: Same.
    gradcam_dict['vector_output_layer_name']: Same.
    gradcam_dict['ideal_activation']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    gradcam_dict = {
        CLASS_ACTIVATIONS_KEY:
            dataset_object.variables[CLASS_ACTIVATIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        ACTIVATION_LAYER_KEY:
            str(getattr(dataset_object, ACTIVATION_LAYER_KEY)),
        VECTOR_OUT_LAYER_KEY:
            str(getattr(dataset_object, VECTOR_OUT_LAYER_KEY)),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    dataset_object.close()
    return gradcam_dict
