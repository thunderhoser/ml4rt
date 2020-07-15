"""Methods for running backwards optimization and reading/writing results."""

import numpy
import netCDF4
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_ITERATIONS = 1000
DEFAULT_L2_WEIGHT = 0.001

INITIAL_PREDICTORS_KEY = 'initial_predictor_matrix'
FINAL_PREDICTORS_KEY = 'final_predictor_matrix'
INITIAL_ACTIVATION_KEY = 'initial_activation'
FINAL_ACTIVATION_KEY = 'final_activation'

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
SCALAR_PREDICTOR_DIM_KEY = 'scalar_predictor'
VECTOR_PREDICTOR_DIM_KEY = 'vector_predictor'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

MODEL_FILE_KEY = 'model_file_name'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
NUM_ITERATIONS_KEY = 'num_iterations'
LEARNING_RATE_KEY = 'learning_rate'
L2_WEIGHT_KEY = 'l2_weight'

EXAMPLE_IDS_KEY = 'example_id_strings'
INIT_SCALAR_PREDICTORS_KEY = 'init_scalar_predictor_matrix'
FINAL_SCALAR_PREDICTORS_KEY = 'final_scalar_predictor_matrix'
INIT_VECTOR_PREDICTORS_KEY = 'init_vector_predictor_matrix'
FINAL_VECTOR_PREDICTORS_KEY = 'final_vector_predictor_matrix'
INITIAL_ACTIVATIONS_KEY = 'initial_activations'
FINAL_ACTIVATIONS_KEY = 'final_activations'


def _do_gradient_descent(
        model_object, activation_tensor, loss_tensor, init_function_or_matrix,
        num_iterations, learning_rate, l2_weight):
    """Nitty-gritty part of backwards optimization.

    :param model_object: See doc for `optimize_input_for_neuron`.
    :param activation_tensor: Keras tensor defining activation of relevant model
        part (single neuron or otherwise).
    :param loss_tensor: Keras tensor defining loss (to be minimized by adjusting
        input).
    :param init_function_or_matrix: See doc for `optimize_input_for_neuron`.
    :param num_iterations: See doc for `check_metadata`.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :return: result_dict: Dictionary with the following keys.
    result_dict['initial_predictor_matrix']: numpy array with predictor matrix
        before optimization.
    result_dict['final_predictor_matrix']: numpy array with predictor matrix
        after optimization.
    result_dict['initial_activation']: Activation of relevant model part (single
        neuron or otherwise) before optimization.
    result_dict['final_activation']: Same but after optimization.
    """

    # Create initial predictor matrix.
    model_input_tensor = model_object.input

    if isinstance(init_function_or_matrix, numpy.ndarray):
        initial_predictor_matrix = init_function_or_matrix
    else:
        dimensions = numpy.array(
            [1] + model_input_tensor.get_shape().as_list()[1:], dtype=int
        )
        initial_predictor_matrix = init_function_or_matrix(dimensions)

    if initial_predictor_matrix.shape[0] != 1:
        initial_predictor_matrix = numpy.expand_dims(
            initial_predictor_matrix, axis=0
        )

    final_predictor_matrix = initial_predictor_matrix + 0.

    # Create gradient tensor.
    if l2_weight > 0:
        difference_matrix = (
            model_input_tensor[0, ...] - final_predictor_matrix[0, ...]
        )
        loss_tensor += l2_weight * K.mean(difference_matrix ** 2)

    gradient_tensor = K.gradients(loss_tensor, [model_input_tensor])[0]
    gradient_tensor /= K.maximum(
        K.sqrt(K.mean(gradient_tensor ** 2)),
        K.epsilon()
    )

    # Create main function (from input to activation, loss, and gradient).
    main_bwo_function = K.function(
        [model_input_tensor, K.learning_phase()],
        [activation_tensor, loss_tensor, gradient_tensor]
    )

    # Do gradient descent.
    initial_activation = None
    these_outputs = None

    for j in range(num_iterations):
        these_outputs = main_bwo_function([final_predictor_matrix, 0])
        if j == 0:
            initial_activation = these_outputs[0][0]

        if numpy.mod(j, 100) == 0:
            print((
                'Loss after {0:d} of {1:d} iterations = {2:.2e} ... '
                'activation = {3:.2e}'
            ).format(
                j, num_iterations, these_outputs[1], these_outputs[0][0]
            ))

        final_predictor_matrix -= these_outputs[2] * learning_rate

    final_activation = these_outputs[0][0]

    print((
        'Loss after {0:d} iterations = {1:.2e} ... activation = {2:.2e}'
    ).format(
        num_iterations, these_outputs[1], final_activation
    ))

    return {
        INITIAL_PREDICTORS_KEY: initial_predictor_matrix,
        FINAL_PREDICTORS_KEY: final_predictor_matrix,
        INITIAL_ACTIVATION_KEY: initial_activation,
        FINAL_ACTIVATION_KEY: final_activation
    }


def check_metadata(
        layer_name, neuron_indices, ideal_activation, num_iterations,
        learning_rate, l2_weight):
    """Checks metadata for errors.

    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension is the batch dimension, which always has length
        `None` in Keras.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (neuron_activation - ideal_activation)**2.
    :param num_iterations: Number of iterations for gradient descent.
    :param learning_rate: Learning rate for gradient descent.
    :param l2_weight: L2 weight (penalty for difference between initial and
        final predictor matrix) in loss function.
    """

    error_checking.assert_is_string(layer_name)

    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

    error_checking.assert_is_not_nan(ideal_activation)

    error_checking.assert_is_integer(num_iterations)
    error_checking.assert_is_greater(num_iterations, 0)

    error_checking.assert_is_greater(learning_rate, 0.)
    error_checking.assert_is_less_than(learning_rate, 1.)

    error_checking.assert_is_geq(l2_weight, 0.)


def create_climo_initializer(model_metadata_dict):
    """Creates climo initializer (one that sets all predictor values to climo).

    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :return: init_function: Function handle (see below).
    """

    def init_function(dimensions):
        """Creates starting point for backwards optimization.

        Specifically, sets all predictor values to climatological mean.

        :param dimensions: 1-D numpy array with dimensions of predictor matrix.
        :return: initial_predictor_matrix: numpy array with climatological
            predictor values.
        """

        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        norm_type_string = (
            generator_option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY]
        )

        if norm_type_string == normalization.Z_SCORE_NORM_STRING:
            return numpy.full(dimensions, 0.)

        min_normalized_value = (
            generator_option_dict[neural_net.PREDICTOR_MIN_NORM_VALUE_KEY]
        )
        max_normalized_value = (
            generator_option_dict[neural_net.PREDICTOR_MAX_NORM_VALUE_KEY]
        )

        return numpy.full(
            dimensions, 0.5 * (min_normalized_value + max_normalized_value)
        )

    return init_function


def optimize_input_for_neuron(
        model_object, init_function_or_matrix, layer_name, neuron_indices,
        ideal_activation, num_iterations=DEFAULT_NUM_ITERATIONS,
        learning_rate=DEFAULT_LEARNING_RATE, l2_weight=DEFAULT_L2_WEIGHT):
    """Optimizes input for activation of one neuron.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param init_function_or_matrix: Function or numpy array used as starting
        point.  If function, must have the following format...

        Input: dimensions: 1-D numpy array with dimensions of predictor matrix.
        Output: predictor_matrix: numpy array of predictors, formatted in the
        same way as for training and inference, but containing only one example.

        If numpy array, must look the same as `predictor_matrix` (see above).

    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :return: result_dict: See doc for `_do_gradient_descent`.
    """

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, num_iterations=num_iterations,
        learning_rate=learning_rate, l2_weight=l2_weight
    )

    activation_tensor = None

    for k in neuron_indices[::-1]:
        if activation_tensor is None:
            activation_tensor = (
                model_object.get_layer(name=layer_name).output[..., k]
            )
        else:
            activation_tensor = activation_tensor[..., k]

    loss_tensor = (activation_tensor - ideal_activation) ** 2

    return _do_gradient_descent(
        model_object=model_object,
        activation_tensor=activation_tensor, loss_tensor=loss_tensor,
        init_function_or_matrix=init_function_or_matrix,
        num_iterations=num_iterations, learning_rate=learning_rate,
        l2_weight=l2_weight
    )


def write_standard_file(
        netcdf_file_name, init_scalar_predictor_matrix,
        final_scalar_predictor_matrix, init_vector_predictor_matrix,
        final_vector_predictor_matrix, initial_activations, final_activations,
        example_id_strings, model_file_name, layer_name, neuron_indices,
        ideal_activation, num_iterations, learning_rate, l2_weight):
    """Writes standard (per-example) backwards-optimization results to file.

    E = number of examples
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors

    :param netcdf_file_name: Path to output file.
    :param init_scalar_predictor_matrix: numpy array (E x P_s) of initial
        predictor values.
    :param final_scalar_predictor_matrix: Same but with final values.
    :param init_vector_predictor_matrix: numpy array (E x H x P_v) of initial
        predictor values.
    :param final_vector_predictor_matrix: Same but with final values.
    :param initial_activations: length-E numpy array of initial activations,
        before optimization.
    :param final_activations: Same but with final activations, after
        optimization.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    """

    # Check input args.
    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, num_iterations=num_iterations,
        learning_rate=learning_rate, l2_weight=l2_weight
    )

    error_checking.assert_is_numpy_array_without_nan(
        init_scalar_predictor_matrix
    )
    error_checking.assert_is_numpy_array(
        init_scalar_predictor_matrix, num_dimensions=2
    )

    error_checking.assert_is_numpy_array_without_nan(
        final_scalar_predictor_matrix
    )
    error_checking.assert_is_numpy_array(
        final_scalar_predictor_matrix,
        exact_dimensions=
        numpy.array(init_scalar_predictor_matrix.shape, dtype=int)
    )

    error_checking.assert_is_numpy_array_without_nan(
        init_vector_predictor_matrix
    )
    error_checking.assert_is_numpy_array(
        init_vector_predictor_matrix, num_dimensions=3
    )

    error_checking.assert_is_numpy_array_without_nan(
        final_vector_predictor_matrix
    )
    error_checking.assert_is_numpy_array(
        final_vector_predictor_matrix,
        exact_dimensions=
        numpy.array(init_vector_predictor_matrix.shape, dtype=int)
    )

    num_examples = init_vector_predictor_matrix.shape[0]
    expected_dim = numpy.array([num_examples], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(initial_activations)
    error_checking.assert_is_numpy_array(
        initial_activations, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(final_activations)
    error_checking.assert_is_numpy_array(
        final_activations, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(LAYER_NAME_KEY, layer_name)
    dataset_object.setncattr(NEURON_INDICES_KEY, neuron_indices)
    dataset_object.setncattr(IDEAL_ACTIVATION_KEY, ideal_activation)
    dataset_object.setncattr(NUM_ITERATIONS_KEY, num_iterations)
    dataset_object.setncattr(LEARNING_RATE_KEY, learning_rate)
    dataset_object.setncattr(L2_WEIGHT_KEY, l2_weight)

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        SCALAR_PREDICTOR_DIM_KEY, init_scalar_predictor_matrix.shape[-1]
    )
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, init_vector_predictor_matrix.shape[1]
    )
    dataset_object.createDimension(
        VECTOR_PREDICTOR_DIM_KEY, init_vector_predictor_matrix.shape[2]
    )

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

    if init_scalar_predictor_matrix.size > 0:
        these_dim = (EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY)

        dataset_object.createVariable(
            INIT_SCALAR_PREDICTORS_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[INIT_SCALAR_PREDICTORS_KEY][:] = (
            init_scalar_predictor_matrix
        )

        dataset_object.createVariable(
            FINAL_SCALAR_PREDICTORS_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[FINAL_SCALAR_PREDICTORS_KEY][:] = (
            final_scalar_predictor_matrix
        )

    if init_vector_predictor_matrix.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_PREDICTOR_DIM_KEY
        )

        dataset_object.createVariable(
            INIT_VECTOR_PREDICTORS_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[INIT_VECTOR_PREDICTORS_KEY][:] = (
            init_vector_predictor_matrix
        )

        dataset_object.createVariable(
            FINAL_VECTOR_PREDICTORS_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[FINAL_VECTOR_PREDICTORS_KEY][:] = (
            final_vector_predictor_matrix
        )

    dataset_object.createVariable(
        INITIAL_ACTIVATIONS_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INITIAL_ACTIVATIONS_KEY][:] = initial_activations

    dataset_object.createVariable(
        FINAL_ACTIVATIONS_KEY, datatype=numpy.float32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[FINAL_ACTIVATIONS_KEY][:] = initial_activations

    dataset_object.close()


def read_standard_file(netcdf_file_name):
    """Reads standard (per-example) backwards-optimization results from. file.

    :param netcdf_file_name: Path to input file.
    :return: bwo_dict: Dictionary with the following keys.
    bwo_dict['init_scalar_predictor_matrix']: See doc for `write_standard_file`.
    bwo_dict['final_scalar_predictor_matrix']: Same.
    bwo_dict['init_vector_predictor_matrix']: Same.
    bwo_dict['final_vector_predictor_matrix']: Same.
    bwo_dict['initial_activations']: Same.
    bwo_dict['final_activations']: Same.
    bwo_dict['example_id_strings']: Same.
    bwo_dict['model_file_name']: Same.
    bwo_dict['layer_name']: Same.
    bwo_dict['neuron_indices']: Same.
    bwo_dict['ideal_activation']: Same.
    bwo_dict['num_iterations']: Same.
    bwo_dict['learning_rate']: Same.
    bwo_dict['l2_weight']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    bwo_dict = {
        INITIAL_ACTIVATIONS_KEY:
            dataset_object.variables[INITIAL_ACTIVATIONS_KEY][:],
        FINAL_ACTIVATIONS_KEY:
            dataset_object.variables[FINAL_ACTIVATIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY),
        NUM_ITERATIONS_KEY:
            int(numpy.round(getattr(dataset_object, NUM_ITERATIONS_KEY))),
        LEARNING_RATE_KEY: getattr(dataset_object, LEARNING_RATE_KEY),
        L2_WEIGHT_KEY: getattr(dataset_object, L2_WEIGHT_KEY)
    }

    num_examples = dataset_object.dimensions[EXAMPLE_DIMENSION_KEY].size
    num_scalar_predictors = (
        dataset_object.dimensions[SCALAR_PREDICTOR_DIM_KEY].size
    )
    num_vector_predictors = (
        dataset_object.dimensions[VECTOR_PREDICTOR_DIM_KEY].size
    )
    num_heights = dataset_object.dimensions[HEIGHT_DIMENSION_KEY].size

    if INIT_SCALAR_PREDICTORS_KEY in dataset_object.variables:
        bwo_dict[INIT_SCALAR_PREDICTORS_KEY] = (
            dataset_object.variables[INIT_SCALAR_PREDICTORS_KEY][:]
        )
        bwo_dict[FINAL_SCALAR_PREDICTORS_KEY] = (
            dataset_object.variables[FINAL_SCALAR_PREDICTORS_KEY][:]
        )
    else:
        these_dim = (num_examples, num_scalar_predictors)
        bwo_dict[INIT_SCALAR_PREDICTORS_KEY] = numpy.full(these_dim, 0.)
        bwo_dict[FINAL_SCALAR_PREDICTORS_KEY] = numpy.full(these_dim, 0.)

    if INIT_VECTOR_PREDICTORS_KEY in dataset_object.variables:
        bwo_dict[INIT_VECTOR_PREDICTORS_KEY] = (
            dataset_object.variables[INIT_VECTOR_PREDICTORS_KEY][:]
        )
        bwo_dict[FINAL_VECTOR_PREDICTORS_KEY] = (
            dataset_object.variables[FINAL_VECTOR_PREDICTORS_KEY][:]
        )
    else:
        these_dim = (num_examples, num_heights, num_vector_predictors)
        bwo_dict[INIT_VECTOR_PREDICTORS_KEY] = numpy.full(these_dim, 0.)
        bwo_dict[FINAL_VECTOR_PREDICTORS_KEY] = numpy.full(these_dim, 0.)

    dataset_object.close()
    return bwo_dict
