"""Methods for building, training, and applying neural nets."""

import numpy
import keras
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import architecture_utils
from ml4rt.io import example_io
from ml4rt.utils import normalization

DEFAULT_L1_WEIGHT = 0.
DEFAULT_L2_WEIGHT = 0.001

DEFAULT_CONV_LAYER_CHANNEL_NUMS = numpy.array([80, 80, 80, 3], dtype=int)
DEFAULT_CONV_LAYER_DROPOUT_RATES = numpy.array([0.5, 0.5, 0.5, numpy.nan])
DEFAULT_DENSE_NEURON_NUMS_FOR_CNN = numpy.array([409, 29, 2], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES_FOR_CNN = numpy.array([0.5, 0.5, numpy.nan])

# TODO(thunderhoser): Probably want to change this.
DEFAULT_DENSE_NEURON_NUMS_FOR_DNN = numpy.array([1000, 409, 29, 2], dtype=int)
DEFAULT_DENSE_DROPOUT_RATES_FOR_DNN = numpy.array([0.5, 0.5, 0.5, numpy.nan])

DEFAULT_INNER_ACTIV_FUNCTION_NAME = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_INNER_ACTIV_FUNCTION_ALPHA = 0.2
DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME = architecture_utils.RELU_FUNCTION_STRING
DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA = 0.

EXAMPLE_DIRECTORY_KEY = 'example_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
PREDICTOR_NAMES_KEY = 'predictor_names'
TARGET_NAMES_KEY = 'target_names'
FIRST_TIME_KEY = 'first_time_unix_sec'
LAST_TIME_KEY = 'last_time_unix_sec'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
PREDICTOR_NORM_TYPE_KEY = 'predictor_norm_type_string'
PREDICTOR_MIN_NORM_VALUE_KEY = 'predictor_min_norm_value'
PREDICTOR_MAX_NORM_VALUE_KEY = 'predictor_max_norm_value'
TARGET_NORM_TYPE_KEY = 'target_norm_type_string'
TARGET_MIN_NORM_VALUE_KEY = 'target_min_norm_value'
TARGET_MAX_NORM_VALUE_KEY = 'target_max_norm_value'

DEFAULT_OPTION_DICT = {
    PREDICTOR_NAMES_KEY: (
        example_io.SCALAR_PREDICTOR_NAMES_KEY +
        example_io.VECTOR_PREDICTOR_NAMES_KEY
    ),
    TARGET_NAMES_KEY: (
        example_io.SCALAR_TARGET_NAMES_KEY + example_io.VECTOR_TARGET_NAMES_KEY
    ),
    PREDICTOR_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING,
    PREDICTOR_MIN_NORM_VALUE_KEY: None,
    PREDICTOR_MAX_NORM_VALUE_KEY: None,
    TARGET_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    TARGET_MIN_NORM_VALUE_KEY: 0.,
    TARGET_MAX_NORM_VALUE_KEY: 1.
}


def _check_generator_args(option_dict):
    """Error-checks input arguments for generator.

    :param option_dict: See doc for any generator in this file.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 32)
    error_checking.assert_is_numpy_array(
        option_dict[PREDICTOR_NAMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        option_dict[TARGET_NAMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_string(option_dict[PREDICTOR_NORM_TYPE_KEY])
    error_checking.assert_is_string(option_dict[TARGET_NORM_TYPE_KEY])

    return option_dict


def _make_cnn_predictor_matrix(example_dict):
    """Makes predictor matrix for CNN.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: predictor_matrix: See output doc for `cnn_generator`.
    """

    num_heights = len(example_dict[example_io.HEIGHTS_KEY])

    scalar_predictor_matrix = example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    scalar_predictor_matrix = numpy.expand_dims(scalar_predictor_matrix, axis=1)
    scalar_predictor_matrix = numpy.repeat(
        scalar_predictor_matrix, repeats=num_heights, axis=1
    )

    return numpy.concatenate((
        example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY],
        scalar_predictor_matrix
    ), axis=-1)


def _make_dense_net_predictor_matrix(example_dict):
    """Makes predictor matrix for dense neural net.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: predictor_matrix: See output doc for `dense_net_generator`.
    """

    vector_predictor_matrix = example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    num_examples = vector_predictor_matrix.shape[0]
    num_heights = vector_predictor_matrix.shape[1]
    num_fields = vector_predictor_matrix.shape[2]

    vector_predictor_matrix = numpy.reshape(
        vector_predictor_matrix, (num_examples, num_heights * num_fields),
        order='F'
    )

    return numpy.concatenate((
        vector_predictor_matrix,
        example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    ), axis=-1)


def _make_dense_net_target_matrix(example_dict):
    """Makes target matrix for dense neural net.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: target_matrix: See output doc for `dense_net_generator`.
    """

    vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]
    num_examples = vector_target_matrix.shape[0]
    num_heights = vector_target_matrix.shape[1]
    num_fields = vector_target_matrix.shape[2]

    vector_target_matrix = numpy.reshape(
        vector_target_matrix, (num_examples, num_heights * num_fields),
        order='F'
    )

    return numpy.concatenate((
        vector_target_matrix,
        example_dict[example_io.SCALAR_TARGET_VALS_KEY]
    ), axis=-1)


def _find_example_files(
        example_dir_name, first_time_unix_sec, last_time_unix_sec,
        test_mode=False):
    """Finds example files.

    :param example_dir_name: See doc for `cnn_generator`.
    :param first_time_unix_sec: Same.
    :param last_time_unix_sec: Same.
    :param test_mode: Leave this alone.
    """

    start_year = int(
        time_conversion.unix_sec_to_string(first_time_unix_sec, '%Y')
    )
    end_year = int(
        time_conversion.unix_sec_to_string(last_time_unix_sec, '%Y')
    )
    years = numpy.linspace(
        start_year, end_year, num=end_year - start_year + 1, dtype=int
    )
    example_file_names = [
        example_io.find_file(
            example_dir_name=example_dir_name, year=y,
            raise_error_if_missing=not test_mode
        )
        for y in years
    ]

    return example_file_names


def _read_file_for_generator(
        example_file_name, num_examples_to_keep, first_time_unix_sec,
        last_time_unix_sec, field_names, normalization_file_name,
        predictor_norm_type_string, predictor_min_norm_value,
        predictor_max_norm_value, target_norm_type_string,
        target_min_norm_value, target_max_norm_value):
    """Reads one file for generator.

    :param example_file_name: Path to input file (will be read by
        `example_io.read_file`).
    :param num_examples_to_keep: Number of examples to keep.
    :param first_time_unix_sec: See doc for `cnn_generator` or
        `dense_net_generator`.
    :param last_time_unix_sec: Same.
    :param field_names: 1-D list of fields to keep.
    :param normalization_file_name: See doc for `cnn_generator` or
        `dense_net_generator`.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param target_norm_type_string: Same.
    :param target_min_norm_value: Same.
    :param target_max_norm_value: Same.
    :return: example_dict: See doc for `example_io.read_file`.
    """

    print('\nReading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(example_file_name)

    example_dict = example_io.reduce_sample_size(
        example_dict=example_dict, num_examples_to_keep=num_examples_to_keep
    )
    example_dict = example_io.subset_by_time(
        example_dict=example_dict,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )
    example_dict = example_io.subset_by_field(
        example_dict=example_dict, field_names=field_names
    )

    print('Applying {0:s} normalization to predictors...'.format(
        predictor_norm_type_string.upper()
    ))
    example_dict = normalization.normalize_data(
        example_dict=example_dict,
        normalization_type_string=predictor_norm_type_string,
        normalization_file_name=normalization_file_name,
        min_normalized_value=predictor_min_norm_value,
        max_normalized_value=predictor_max_norm_value,
        separate_heights=False,
        apply_to_predictors=True, apply_to_targets=False
    )

    print('Applying {0:s} normalization to targets...'.format(
        target_norm_type_string.upper()
    ))
    example_dict = normalization.normalize_data(
        example_dict=example_dict,
        normalization_type_string=target_norm_type_string,
        normalization_file_name=normalization_file_name,
        min_normalized_value=target_min_norm_value,
        max_normalized_value=target_max_norm_value,
        separate_heights=False,
        apply_to_predictors=False, apply_to_targets=True
    )

    return example_dict


def make_cnn(
        num_heights, num_input_channels,
        conv_layer_channel_nums=DEFAULT_CONV_LAYER_CHANNEL_NUMS,
        conv_layer_dropout_rates=DEFAULT_CONV_LAYER_DROPOUT_RATES,
        dense_layer_neuron_nums=DEFAULT_DENSE_NEURON_NUMS_FOR_CNN,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES_FOR_CNN,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Makes CNN (convolutional neural net).

    This method only sets up the architecture, loss function, and optimizer,
    then compiles the model.  This method does *not* train the model.

    C = number of convolutional layers
    D = number of dense layers

    :param num_heights: Number of height levels.
    :param num_input_channels: Number of input channels.
    :param conv_layer_channel_nums: length-C numpy array with number of channels
        (filters) produced by each conv layer.  The last value in the array,
        conv_layer_channel_nums[-1], is the number of output channels (profiles
        to be predicted).
    :param conv_layer_dropout_rates: length-C numpy array with dropout rate for
        each conv layer.  Use NaN if you do not want dropout for a particular
        layer.
    :param dense_layer_neuron_nums: length-D numpy array with number of neurons
        (features) produced by each dense layer.  The last value in the array,
        dense_layer_neuron_nums[-1], is the number of output scalars (to be
        predicted).
    :param dense_layer_dropout_rates: length-D numpy array with dropout rate for
        each dense layer.  Use NaN if you do not want dropout for a particular
        layer.
    :param inner_activ_function_name: Name of activation function for all inner
        (non-output) layers.  Must be accepted by ``.
    :param inner_activ_function_alpha: Alpha (slope parameter) for activation
        function for all inner layers.  Applies only to ReLU and eLU.
    :param output_activ_function_name: Same as `inner_activ_function_name` but
        for output layers (profiles and scalars).
    :param output_activ_function_alpha: Same as `inner_activ_function_alpha` but
        for output layers (profiles and scalars).
    :param l1_weight: Weight for L_1 regularization.
    :param l2_weight: Weight for L_2 regularization.
    :param use_batch_normalization: Boolean flag.  If True, will use batch
        normalization after each inner (non-output) layer.
    :return: model_object: Untrained instance of `keras.models.Model`.
    """

    # TODO(thunderhoser): Check input args.
    # TODO(thunderhoser): Filter size needs to be an option.

    input_layer_object = keras.layers.Input(
        shape=(num_heights, num_input_channels)
    )
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_conv_layers = len(conv_layer_channel_nums)
    conv_output_layer_object = None
    dense_input_layer_object = None

    for i in range(num_conv_layers):
        if conv_output_layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = conv_output_layer_object

        if i == num_conv_layers - 1:
            dense_input_layer_object = conv_output_layer_object

        conv_output_layer_object = architecture_utils.get_1d_conv_layer(
            num_kernel_rows=5, num_rows_per_stride=1,
            num_filters=conv_layer_channel_nums[i],
            padding_type_string=architecture_utils.YES_PADDING_STRING,
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_conv_layers - 1:
            conv_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
            )(conv_output_layer_object)
        else:
            conv_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(conv_output_layer_object)

        if conv_layer_dropout_rates[i] > 0:
            conv_output_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=conv_layer_dropout_rates[i]
            )(conv_output_layer_object)

        if use_batch_normalization and i != num_conv_layers - 1:
            conv_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    conv_output_layer_object
                )
            )

    num_dense_layers = len(dense_layer_neuron_nums)
    dense_output_layer_object = architecture_utils.get_flattening_layer()(
        dense_input_layer_object
    )

    for i in range(num_dense_layers):
        dense_output_layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[i]
        )(dense_output_layer_object)

        if i == num_dense_layers - 1:
            dense_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
            )(dense_output_layer_object)
        else:
            dense_output_layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(dense_output_layer_object)

        if dense_layer_dropout_rates[i] > 0:
            dense_output_layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(dense_output_layer_object)

        if use_batch_normalization and i != num_dense_layers - 1:
            dense_output_layer_object = (
                architecture_utils.get_batch_norm_layer()(
                    dense_output_layer_object
                )
            )

    model_object = keras.models.Model(
        inputs=input_layer_object,
        outputs=[conv_output_layer_object, dense_output_layer_object]
    )

    # TODO(thunderhoser): Add bias to metrics.
    model_object.compile(
        loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),
        metrics=['mae']
    )

    model_object.summary()
    return model_object


def make_dense_net(
        num_inputs,
        dense_layer_neuron_nums=DEFAULT_DENSE_NEURON_NUMS_FOR_DNN,
        dense_layer_dropout_rates=DEFAULT_DENSE_DROPOUT_RATES_FOR_DNN,
        inner_activ_function_name=DEFAULT_INNER_ACTIV_FUNCTION_NAME,
        inner_activ_function_alpha=DEFAULT_INNER_ACTIV_FUNCTION_ALPHA,
        output_activ_function_name=DEFAULT_OUTPUT_ACTIV_FUNCTION_NAME,
        output_activ_function_alpha=DEFAULT_OUTPUT_ACTIV_FUNCTION_ALPHA,
        l1_weight=DEFAULT_L1_WEIGHT, l2_weight=DEFAULT_L2_WEIGHT,
        use_batch_normalization=True):
    """Makes dense (fully connected) neural net.

    :param num_inputs: Number of input variables (predictors).
    :param dense_layer_neuron_nums: See doc for `make_cnn`.
    :param dense_layer_dropout_rates: Same.
    :param inner_activ_function_name: Same.
    :param inner_activ_function_alpha: Same.
    :param output_activ_function_name: Same.
    :param output_activ_function_alpha: Same.
    :param l1_weight: Same.
    :param l2_weight: Same.
    :param use_batch_normalization: Same.
    :return: model_object: Same.
    """

    # TODO(thunderhoser): Check input args.

    input_layer_object = keras.layers.Input(shape=(num_inputs,))
    regularizer_object = architecture_utils.get_weight_regularizer(
        l1_weight=l1_weight, l2_weight=l2_weight
    )

    num_dense_layers = len(dense_layer_neuron_nums)
    layer_object = None

    for i in range(num_dense_layers):
        if layer_object is None:
            this_input_layer_object = input_layer_object
        else:
            this_input_layer_object = layer_object

        layer_object = architecture_utils.get_dense_layer(
            num_output_units=dense_layer_neuron_nums[i],
            weight_regularizer=regularizer_object
        )(this_input_layer_object)

        if i == num_dense_layers - 1:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=output_activ_function_name,
                alpha_for_relu=output_activ_function_alpha,
                alpha_for_elu=output_activ_function_alpha
            )(layer_object)
        else:
            layer_object = architecture_utils.get_activation_layer(
                activation_function_string=inner_activ_function_name,
                alpha_for_relu=inner_activ_function_alpha,
                alpha_for_elu=inner_activ_function_alpha
            )(layer_object)

        if dense_layer_dropout_rates[i] > 0:
            layer_object = architecture_utils.get_dropout_layer(
                dropout_fraction=dense_layer_dropout_rates[i]
            )(layer_object)

        if use_batch_normalization and i != num_dense_layers - 1:
            layer_object = architecture_utils.get_batch_norm_layer()(
                layer_object
            )

    model_object = keras.models.Model(
        inputs=input_layer_object, outputs=layer_object
    )

    # TODO(thunderhoser): Add bias to metrics.
    model_object.compile(
        loss=keras.losses.mse, optimizer=keras.optimizers.Adam(),
        metrics=['mae']
    )

    model_object.summary()
    return model_object


def cnn_generator(option_dict):
    """Generates examples for CNN.

    E = number of examples per batch (batch size)
    H = number of heights
    P = number of predictor variables (channels)
    T_v = number of vector target variables (channels)
    T_s = number of scalar target variables

    :param option_dict: Dictionary with the following keys.
    option_dict['example_dir_name']: Name of directory with example files.
        Files therein will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    option_dict['num_examples_per_batch']: Batch size.
    option_dict['predictor_names']: 1-D list with names of predictor variables
        (valid names listed in example_io.py).
    option_dict['target_names']: Same but for target variables.
    option_dict['first_time_unix_sec']: Start time (will not generate examples
        before this time).
    option_dict['last_time_unix_sec']: End time (will not generate examples after
        this time).
    option_dict['normalization_file_name']: File with normalization parameters
        for both predictors and targets (will be read by
        `normalization_params.read_file`).
    option_dict['predictor_norm_type_string']: Normalization type for predictors
        (must be accepted by `normalization._check_normalization_type`).
    option_dict['predictor_min_norm_value']: Minimum normalized value for
        predictors (used only if normalization type is min-max).
    option_dict['predictor_max_norm_value']: Same but max value.
    option_dict['target_norm_type_string']: Normalization type for targets (must
        be accepted by `normalization._check_normalization_type`).
    option_dict['target_min_norm_value']: Minimum normalized value for targets
        (used only if normalization type is min-max).
    option_dict['target_max_norm_value']: Same but max value.

    :return: predictor_matrix: E-by-H-by-P numpy array of predictor values.
    :return: target_list: List with 2 items.
    target_list[0] = vector_target_matrix: numpy array (E x H x T_v) of target
        values.
    target_list[1] = scalar_target_matrix: numpy array (E x T_s) of target
        values.
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    predictor_names = option_dict[PREDICTOR_NAMES_KEY]
    target_names = option_dict[TARGET_NAMES_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = predictor_names +  target_names

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    target_norm_type_string = option_dict[TARGET_NORM_TYPE_KEY]
    target_min_norm_value = option_dict[TARGET_MIN_NORM_VALUE_KEY]
    target_max_norm_value = option_dict[TARGET_MAX_NORM_VALUE_KEY]

    example_file_names = _find_example_files(
        example_dir_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )

    file_index = 0
    num_examples_in_memory = 0

    while True:
        predictor_matrix = None
        vector_target_matrix = None
        scalar_target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            this_example_dict = _read_file_for_generator(
                example_file_name=example_file_names[file_index],
                num_examples_to_keep=
                num_examples_per_batch - num_examples_in_memory,
                first_time_unix_sec=first_time_unix_sec,
                last_time_unix_sec=last_time_unix_sec,
                field_names=all_field_names,
                normalization_file_name=normalization_file_name,
                predictor_norm_type_string=predictor_norm_type_string,
                predictor_min_norm_value=predictor_min_norm_value,
                predictor_max_norm_value=predictor_max_norm_value,
                target_norm_type_string=target_norm_type_string,
                target_min_norm_value=target_min_norm_value,
                target_max_norm_value=target_max_norm_value
            )

            file_index += 1
            num_examples_in_memory += len(
                this_example_dict[example_io.VALID_TIMES_KEY]
            )

            this_predictor_matrix = _make_cnn_predictor_matrix(
                example_dict=this_example_dict
            )
            this_vector_target_matrix = (
                this_example_dict[example_io.VECTOR_TARGET_VALS_KEY]
            )
            this_scalar_target_matrix = (
                this_example_dict[example_io.SCALAR_TARGET_VALS_KEY]
            )

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                vector_target_matrix = this_vector_target_matrix + 0.
                scalar_target_matrix = this_scalar_target_matrix + 0.
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )
                vector_target_matrix = numpy.concatenate(
                    (vector_target_matrix, this_vector_target_matrix), axis=0
                )
                scalar_target_matrix = numpy.concatenate(
                    (scalar_target_matrix, this_scalar_target_matrix), axis=0
                )

        predictor_matrix = predictor_matrix.astype('float32')
        vector_target_matrix = vector_target_matrix.astype('float32')
        scalar_target_matrix = scalar_target_matrix.astype('float32')

        yield (predictor_matrix, [vector_target_matrix, scalar_target_matrix])


def dense_net_generator(option_dict):
    """Generates examples for dense neural net.

    E = number of examples per batch (batch size)
    P = number of predictor variables
    T = number of target variables

    :param option_dict: See doc for `cnn_generator`.
    :return: predictor_matrix: E-by-P numpy array of predictor values.
    :return: target_matrix: E-by-T numpy array of target values.
    """

    option_dict = _check_generator_args(option_dict)

    example_dir_name = option_dict[EXAMPLE_DIRECTORY_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    predictor_names = option_dict[PREDICTOR_NAMES_KEY]
    target_names = option_dict[TARGET_NAMES_KEY]
    first_time_unix_sec = option_dict[FIRST_TIME_KEY]
    last_time_unix_sec = option_dict[LAST_TIME_KEY]

    all_field_names = predictor_names +  target_names

    normalization_file_name = option_dict[NORMALIZATION_FILE_KEY]
    predictor_norm_type_string = option_dict[PREDICTOR_NORM_TYPE_KEY]
    predictor_min_norm_value = option_dict[PREDICTOR_MIN_NORM_VALUE_KEY]
    predictor_max_norm_value = option_dict[PREDICTOR_MAX_NORM_VALUE_KEY]
    target_norm_type_string = option_dict[TARGET_NORM_TYPE_KEY]
    target_min_norm_value = option_dict[TARGET_MIN_NORM_VALUE_KEY]
    target_max_norm_value = option_dict[TARGET_MAX_NORM_VALUE_KEY]

    example_file_names = _find_example_files(
        example_dir_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )

    file_index = 0
    num_examples_in_memory = 0

    while True:
        predictor_matrix = None
        target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            this_example_dict = _read_file_for_generator(
                example_file_name=example_file_names[file_index],
                num_examples_to_keep=
                num_examples_per_batch - num_examples_in_memory,
                first_time_unix_sec=first_time_unix_sec,
                last_time_unix_sec=last_time_unix_sec,
                field_names=all_field_names,
                normalization_file_name=normalization_file_name,
                predictor_norm_type_string=predictor_norm_type_string,
                predictor_min_norm_value=predictor_min_norm_value,
                predictor_max_norm_value=predictor_max_norm_value,
                target_norm_type_string=target_norm_type_string,
                target_min_norm_value=target_min_norm_value,
                target_max_norm_value=target_max_norm_value
            )

            file_index += 1
            num_examples_in_memory += len(
                this_example_dict[example_io.VALID_TIMES_KEY]
            )

            this_predictor_matrix = _make_dense_net_predictor_matrix(
                example_dict=this_example_dict
            )
            this_target_matrix = _make_dense_net_target_matrix(
                example_dict=this_example_dict
            )

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
                target_matrix = this_target_matrix + 0.
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=0
                )
                target_matrix = numpy.concatenate(
                    (target_matrix, this_target_matrix), axis=0
                )

        yield (
            predictor_matrix.astype('float32'),
            target_matrix.astype('float32')
        )
