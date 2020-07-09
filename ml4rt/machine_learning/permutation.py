"""IO and helper methods for permutation-based importance test."""

import copy
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.machine_learning import neural_net

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

DEFAULT_NUM_BOOTSTRAP_REPS = 1000

PREDICTORS_KEY = 'predictor_matrix'
PERMUTED_FLAGS_KEY = 'permuted_flag_matrix'
PERMUTED_CHANNELS_KEY = 'permuted_channel_indices'
PERMUTED_HEIGHTS_KEY = 'permuted_height_indices'
PERMUTED_COSTS_KEY = 'permuted_cost_matrix'
DEPERMUTED_CHANNELS_KEY = 'depermuted_channel_indices'
DEPERMUTED_HEIGHTS_KEY = 'depermuted_height_indices'
DEPERMUTED_COSTS_KEY = 'depermuted_cost_matrix'

ORIGINAL_COST_KEY = 'orig_cost_estimates'
BEST_PREDICTORS_KEY = 'best_predictor_names'
BEST_HEIGHTS_KEY = 'best_heights_m_agl'
BEST_COSTS_KEY = 'best_cost_matrix'
STEP1_PREDICTORS_KEY = 'step1_predictor_names'
STEP1_HEIGHTS_KEY = 'step1_heights_m_agl'
STEP1_COSTS_KEY = 'step1_cost_matrix'


def _permute_values(
        predictor_matrix, channel_index, height_index=None,
        permuted_value_matrix=None):
    """Permutes values of one predictor variable across all examples.

    :param predictor_matrix: See doc for `run_forward_test_one_step`.
    :param channel_index: Will permute values only in this channel.
    :param height_index:
        [ignored if None or if `predictor_matrix` is 2-D rather than 3-D]
        Will permute values only at this height.
    :param permuted_value_matrix: numpy array of permuted values to replace
        original ones.  This matrix must have the same shape as the submatrix
        being replaced (predictor_matrix[..., channel_index] or
        predictor_matrix[..., height_index, channel_index]).  If None, values
        will be permuted on the fly.
    :return: predictor_matrix: Same as input but with desired values permuted.
    :return: permuted_value_matrix: See input doc.  If input was None, this will
        be a new array created on the fly.  If input was specified, this will be
        the same as input.
    """

    if permuted_value_matrix is None:
        random_indices = numpy.random.permutation(predictor_matrix.shape[0])

        if height_index is None:
            permuted_value_matrix = (
                predictor_matrix[..., channel_index][random_indices, ...]
            )
            predictor_matrix[..., channel_index] = permuted_value_matrix
        else:
            permuted_value_matrix = (
                predictor_matrix[..., height_index, channel_index][
                    random_indices, ...
                ]
            )

            predictor_matrix[..., height_index, channel_index] = (
                permuted_value_matrix
            )
    else:
        if height_index is None:
            predictor_matrix[..., channel_index] = permuted_value_matrix
        else:
            predictor_matrix[..., height_index, channel_index] = (
                permuted_value_matrix
            )

    return predictor_matrix, permuted_value_matrix


def _depermute_values(
        predictor_matrix, clean_predictor_matrix, channel_index,
        height_index=None):
    """Depermutes (cleans up) values of one predictor variable.

    :param predictor_matrix: See doc for `_permute_values`.
    :param clean_predictor_matrix: Clean version of `predictor_matrix` (with no
        values permuted).
    :param channel_index: See doc for `_permute_values`.
    :param height_index: Same.
    :return: predictor_matrix: Same as input but with desired values depermuted.
    """

    if height_index is None:
        predictor_matrix[..., channel_index] = (
            clean_predictor_matrix[..., channel_index]
        )
    else:
        predictor_matrix[..., height_index, channel_index] = (
            clean_predictor_matrix[..., height_index, channel_index]
        )

    return predictor_matrix


def _bootstrap_cost(target_matrices, prediction_matrices, cost_function,
                    num_replicates):
    """Uses bootstrapping to estimate cost.

    N = number of target matrices

    :param target_matrices: length-N list of numpy arrays containing actual
        (target) values.
    :param prediction_matrices: length-N list of numpy arrays containing
        predicted values.
    :param cost_function: Cost function.  Must be negatively oriented (i.e.,
        lower is better), with the following inputs and outputs.
    Input: target_matrices: See above.
    Input: prediction_matrices: See above.
    Output: cost: Scalar value.

    :param num_replicates: Number of bootstrap replicates (i.e., number of times
        to estimate cost).
    :return: cost_estimates: length-B numpy array of cost estimates, where B =
        number of bootstrap replicates.
    """

    cost_estimates = numpy.full(num_replicates, numpy.nan)

    if num_replicates == 1:
        cost_estimates[0] = cost_function(target_matrices, prediction_matrices)
    else:
        num_examples = target_matrices[0].shape[0]
        example_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )

        for k in range(num_replicates):
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

            cost_estimates[k] = cost_function(
                [t[these_indices, ...] for t in target_matrices],
                [p[these_indices, ...] for p in prediction_matrices]
            )

    print('Average cost estimate over {0:d} replicates = {1:f}'.format(
        num_replicates, numpy.mean(cost_estimates)
    ))

    return cost_estimates


def _check_args_one_step(
        predictor_matrix, permuted_flag_matrix, shuffle_profiles_together,
        num_bootstrap_reps):
    """Checks input args for `run_*_test_one_step`.

    :param predictor_matrix: See doc for `run_forward_test_one_step` or
        `run_backwards_test_one_step`.
    :param permuted_flag_matrix: Same.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :return: num_bootstrap_reps: Same as input but maxxed with 1.
    """

    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    num_predictor_dim = len(predictor_matrix.shape)
    error_checking.assert_is_geq(num_predictor_dim, 2)
    error_checking.assert_is_leq(num_predictor_dim, 3)

    error_checking.assert_is_boolean_numpy_array(permuted_flag_matrix)
    these_expected_dim = numpy.array(predictor_matrix.shape[1:], dtype=int)
    error_checking.assert_is_numpy_array(
        permuted_flag_matrix, exact_dimensions=these_expected_dim
    )

    if num_predictor_dim == 2:
        shuffle_profiles_together = True
    error_checking.assert_is_boolean(shuffle_profiles_together)

    error_checking.assert_is_integer(num_bootstrap_reps)
    return numpy.maximum(num_bootstrap_reps, 1)


def run_forward_test_one_step(
        predictor_matrix, target_matrices, prediction_function, cost_function,
        permuted_flag_matrix, shuffle_profiles_together=True,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs one step of the forward permutation test.

    E = number of examples
    H = number of heights
    C = number of channels
    N = number of target matrices

    :param predictor_matrix: numpy array (either E x H x C or E x C) of
        predictor values.
    :param target_matrices: length-N list of numpy arrays, each containing
        target values.
    :param prediction_function: Function with the following inputs and outputs.
    Input: predictor_matrix: See above.
    Output: prediction_matrices: length-N list of numpy arrays, each containing
        predicted values.  prediction_matrices[i] should have the same shape as
        target_matrices[i].

    :param cost_function: See doc for `_bootstrap_cost`.
    :param permuted_flag_matrix: numpy array of Boolean flags, indicating which
        predictors have already been permuted in a previous step.  If
        `predictor_matrix` is E x H x C, this array should be H x C.  If
        `predictor_matrix` is E x C, this array should have length C.
    :param shuffle_profiles_together: Boolean flag.  If True, vertical profiles
        will be shuffled together (i.e., shuffling will be done only along the
        example axis).  If False, all scalar variables will be shuffled
        independently (i.e., shuffling will be done along both the example and
        height axes), so vertical profiles will be destroyed by shuffling.
    :param num_bootstrap_reps: See doc for `_bootstrap_cost`.
    :return: result_dict: Dictionary with the following keys, where P = number
        of permutations done in this step and B = number of bootstrap
        replicates.
    result_dict['predictor_matrix']: Same as input but with more values
        permuted.
    result_dict['permuted_flag_matrix']: Same as input but with more `True`
        flags.
    result_dict['permuted_channel_indices']: length-P numpy array with indices
        of channels permuted.
    result_dict['permuted_height_indices']: length-P numpy array with indices of
        corresponding heights permuted.  This may also be None.
    result_dict['permuted_cost_matrix']: P-by-B numpy array of costs after
        permutation.
    """

    # TODO(thunderhoser): Allow profiles to be shuffled together for 2-D
    # predictor matrix as well.  I probably want to take care of this in a
    # higher-level method.

    num_bootstrap_reps = _check_args_one_step(
        predictor_matrix=predictor_matrix,
        permuted_flag_matrix=permuted_flag_matrix,
        shuffle_profiles_together=shuffle_profiles_together,
        num_bootstrap_reps=num_bootstrap_reps
    )

    num_predictor_dim = len(predictor_matrix.shape)

    # Housekeeping.
    if shuffle_profiles_together:
        num_permutations = numpy.sum(permuted_flag_matrix[0, :] == False)
        permuted_height_indices = None
    else:
        num_permutations = numpy.sum(permuted_flag_matrix == False)
        permuted_height_indices = numpy.full(num_permutations, -1, dtype=int)

    if num_permutations == 0:
        return None

    permuted_channel_indices = numpy.full(num_permutations, -1, dtype=int)
    permuted_cost_matrix = numpy.full(
        (num_permutations, num_bootstrap_reps), numpy.nan
    )

    num_channels = predictor_matrix.shape[-1]
    if shuffle_profiles_together:
        num_heights = 1
    else:
        num_heights = predictor_matrix.shape[-2]

    i = -1
    best_cost = -numpy.inf
    best_channel_index = -1
    best_permuted_value_matrix = None
    if shuffle_profiles_together:
        best_height_index = None
    else:
        best_height_index = -1

    for j in range(num_heights):
        for k in range(num_channels):
            if num_predictor_dim == 3:
                this_flag = permuted_flag_matrix[j, k]
            else:
                this_flag = permuted_flag_matrix[k]

            if this_flag:
                continue

            i += 1
            permuted_channel_indices[i] = k
            if not shuffle_profiles_together:
                permuted_height_indices[i] = j

            log_string = 'Permuting {0:d}th of {1:d} channels'.format(
                k + 1, num_channels
            )

            if shuffle_profiles_together:
                log_string += '...'
            else:
                log_string += ' at {0:d}th of {1:d} heights...'.format(
                    j + 1, num_heights
                )

            print(log_string)

            this_predictor_matrix, this_permuted_value_matrix = _permute_values(
                predictor_matrix=predictor_matrix + 0.,
                channel_index=k,
                height_index=None if shuffle_profiles_together else j
            )
            these_prediction_matrices = prediction_function(
                this_predictor_matrix
            )
            permuted_cost_matrix[i, :] = _bootstrap_cost(
                target_matrices=target_matrices,
                prediction_matrices=these_prediction_matrices,
                cost_function=cost_function, num_replicates=num_bootstrap_reps
            )

            this_average_cost = numpy.mean(permuted_cost_matrix[i, :])
            if this_average_cost < best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_channel_index = k
            best_permuted_value_matrix = this_permuted_value_matrix
            if shuffle_profiles_together:
                best_height_index = j

    predictor_matrix = _permute_values(
        predictor_matrix=predictor_matrix,
        channel_index=best_channel_index,
        height_index=best_height_index,
        permuted_value_matrix=best_permuted_value_matrix
    )[0]

    log_string = 'Best predictor = {0:d}th channel'.format(
        best_channel_index + 1
    )
    if not shuffle_profiles_together:
        log_string += 'at {0:d}th height'.format(best_height_index + 1)

    log_string += '(cost = {0:.4f})'.format(best_cost)
    print(log_string)

    if shuffle_profiles_together:
        permuted_flag_matrix[..., best_channel_index] = True
    else:
        permuted_flag_matrix[best_height_index, best_channel_index] = True

    return {
        PREDICTORS_KEY: predictor_matrix,
        PERMUTED_FLAGS_KEY: permuted_flag_matrix,
        PERMUTED_CHANNELS_KEY: permuted_channel_indices,
        PERMUTED_HEIGHTS_KEY: permuted_height_indices,
        PERMUTED_COSTS_KEY: permuted_cost_matrix,
    }


def run_backwards_test_one_step(
        predictor_matrix, clean_predictor_matrix, target_matrices,
        prediction_function, cost_function, permuted_flag_matrix,
        shuffle_profiles_together=True,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs one step of the backwards permutation test.

    E = number of examples
    H = number of heights
    C = number of channels
    N = number of target matrices

    :param predictor_matrix: See doc for `run_forward_test_one_step`.
    :param clean_predictor_matrix: Clean version of `predictor_matrix` (with no
        values permuted).
    :param target_matrices: See doc for `run_forward_test_one_step`.
    :param prediction_function: Same.
    :param cost_function: Same.
    :param permuted_flag_matrix: Same.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :return: result_dict: Dictionary with the following keys, where P = number
        of depermutations done in this step and B = number of bootstrap
        replicates.
    result_dict['predictor_matrix']: Same as input but with fewer values
        permuted.
    result_dict['permuted_flag_matrix']: Same as input but with more `False`
        flags.
    result_dict['depermuted_channel_indices']: length-P numpy array with indices
        of channels depermuted.
    result_dict['depermuted_height_indices']: length-P numpy array with indices
        of corresponding heights depermuted.  This may also be None.
    result_dict['depermuted_cost_matrix']: P-by-B numpy array of costs after
        depermutation.
    """

    num_bootstrap_reps = _check_args_one_step(
        predictor_matrix=predictor_matrix,
        permuted_flag_matrix=permuted_flag_matrix,
        shuffle_profiles_together=shuffle_profiles_together,
        num_bootstrap_reps=num_bootstrap_reps
    )

    num_predictor_dim = len(predictor_matrix.shape)

    error_checking.assert_is_numpy_array_without_nan(clean_predictor_matrix)
    error_checking.assert_is_numpy_array(
        clean_predictor_matrix,
        exact_dimensions=numpy.array(predictor_matrix, dtype=int)
    )

    # Housekeeping.
    if shuffle_profiles_together:
        num_depermutations = numpy.sum(permuted_flag_matrix[0, :])
        depermuted_height_indices = None
    else:
        num_depermutations = numpy.sum(permuted_flag_matrix)
        depermuted_height_indices = numpy.full(
            num_depermutations, -1, dtype=int
        )

    if num_depermutations == 0:
        return None

    depermuted_channel_indices = numpy.full(num_depermutations, -1, dtype=int)
    depermuted_cost_matrix = numpy.full(
        (num_depermutations, num_bootstrap_reps), numpy.nan
    )

    num_channels = predictor_matrix.shape[-1]
    if shuffle_profiles_together:
        num_heights = 1
    else:
        num_heights = predictor_matrix.shape[-2]

    i = -1
    best_cost = numpy.inf
    best_channel_index = -1
    if shuffle_profiles_together:
        best_height_index = None
    else:
        best_height_index = -1

    for j in range(num_heights):
        for k in range(num_channels):
            if num_predictor_dim == 3:
                this_flag = permuted_flag_matrix[j, k]
            else:
                this_flag = permuted_flag_matrix[k]

            if not this_flag:
                continue

            i += 1
            depermuted_channel_indices[i] = k
            if not shuffle_profiles_together:
                depermuted_height_indices[i] = j

            log_string = 'Depermuting {0:d}th of {1:d} channels'.format(
                k + 1, num_channels
            )

            if shuffle_profiles_together:
                log_string += '...'
            else:
                log_string += ' at {0:d}th of {1:d} heights...'.format(
                    j + 1, num_heights
                )

            print(log_string)

            this_predictor_matrix = _depermute_values(
                predictor_matrix=predictor_matrix + 0.,
                clean_predictor_matrix=clean_predictor_matrix,
                channel_index=k,
                height_index=None if shuffle_profiles_together else j
            )
            these_prediction_matrices = prediction_function(
                this_predictor_matrix
            )
            depermuted_cost_matrix[i, :] = _bootstrap_cost(
                target_matrices=target_matrices,
                prediction_matrices=these_prediction_matrices,
                cost_function=cost_function, num_replicates=num_bootstrap_reps
            )

            this_average_cost = numpy.mean(depermuted_cost_matrix[i, :])
            if this_average_cost > best_cost:
                continue

            best_cost = this_average_cost + 0.
            best_channel_index = k
            if shuffle_profiles_together:
                best_height_index = j

    predictor_matrix = _depermute_values(
        predictor_matrix=predictor_matrix,
        clean_predictor_matrix=clean_predictor_matrix,
        channel_index=best_channel_index, height_index=best_height_index
    )

    log_string = 'Best predictor = {0:d}th channel'.format(
        best_channel_index + 1
    )
    if not shuffle_profiles_together:
        log_string += 'at {0:d}th height'.format(best_height_index + 1)

    log_string += '(cost = {0:.4f})'.format(best_cost)
    print(log_string)

    if shuffle_profiles_together:
        permuted_flag_matrix[..., best_channel_index] = False
    else:
        permuted_flag_matrix[best_height_index, best_channel_index] = False

    return {
        PREDICTORS_KEY: predictor_matrix,
        PERMUTED_FLAGS_KEY: permuted_flag_matrix,
        DEPERMUTED_CHANNELS_KEY: depermuted_channel_indices,
        DEPERMUTED_HEIGHTS_KEY: depermuted_height_indices,
        DEPERMUTED_COSTS_KEY: depermuted_cost_matrix,
    }


def _make_prediction_function(model_object, net_type_string):
    """Creates prediction function for neural net (any type).

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param net_type_string: Type of neural net (must be accepted by
        `neural_net.check_net_type`).
    :return: prediction_function: Function defined below.
    """

    def prediction_function(predictor_matrix):
        """Prediction function itself.

        :param predictor_matrix: See doc for `run_forward_test_one_step`.
        :return: prediction_matrices: 1-D list of numpy arrays, each containing
            predicted values.
        """

        return neural_net.apply_model(
            model_object=model_object, predictor_matrix=predictor_matrix,
            num_examples_per_batch=1000, net_type_string=net_type_string,
            is_loss_constrained_mse=False, verbose=False
        )

    return prediction_function


def _predictor_indices_to_metadata(
        all_predictor_name_matrix, all_height_matrix_m_agl,
        one_step_result_dict):
    """Converts predictor indices to metadata (name and height).

    N = number of permutations or depermutations

    :param all_predictor_name_matrix: See output doc for
        `neural_net.predictors_dict_to_numpy`.
    :param all_height_matrix_m_agl: Same.
    :param one_step_result_dict: Dictionary created by
        `run_forward_test_one_step` or `run_backwards_test_one_step`.
    :return: predictor_names: length-N list of predictor names, in the order
        that they were (de)permuted.
    :return: heights_m_agl: length-N numpy array of corresponding heights
        (metres above ground level).
    """

    if PERMUTED_CHANNELS_KEY in one_step_result_dict:
        channel_indices = one_step_result_dict[PERMUTED_CHANNELS_KEY]
    else:
        channel_indices = one_step_result_dict[DEPERMUTED_CHANNELS_KEY]

    predictor_names = [
        all_predictor_name_matrix[..., k] for k in channel_indices
    ]
    if predictor_names[0].shape:
        predictor_names = [n[0] for n in predictor_names]

    if PERMUTED_HEIGHTS_KEY in one_step_result_dict:
        height_indices = one_step_result_dict[PERMUTED_HEIGHTS_KEY]
    else:
        height_indices = one_step_result_dict[DEPERMUTED_HEIGHTS_KEY]

    if height_indices is None:
        heights_m_agl = numpy.full(len(predictor_names), numpy.nan)
        return predictor_names, heights_m_agl

    heights_m_agl = [
        all_height_matrix_m_agl[j, ...] for j in height_indices
    ]
    if heights_m_agl[0].shape:
        heights_m_agl = [h[0] for h in heights_m_agl]

    heights_m_agl = numpy.array(heights_m_agl)
    return predictor_names, heights_m_agl


def run_forward_test(
        predictor_matrix, target_matrices, model_object, model_metadata_dict,
        cost_function, shuffle_profiles_together=True,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs forward version of permutation test (both single- and multi-pass).

    :param predictor_matrix: See doc for `run_forward_test_one_step`.
    :param target_matrices: Same.
    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param cost_function: See doc for `run_forward_test_one_step`.
    :param shuffle_profiles_together: Same.
    :param num_bootstrap_reps: Same.
    :return: Still need to decide...
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    example_dict = {
        example_io.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_io.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    new_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=predictor_matrix, example_dict=example_dict,
        net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY]
    )
    example_dict.update(new_example_dict)

    predictor_name_matrix, height_matrix_m_agl = (
        neural_net.predictors_dict_to_numpy(
            example_dict=example_dict,
            net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY]
        )[1:]
    )

    error_checking.assert_is_boolean(shuffle_profiles_together)
    error_checking.assert_is_integer(num_bootstrap_reps)
    num_bootstrap_reps = numpy.maximum(num_bootstrap_reps, 1)

    # TODO(thunderhoser): Still need cost functions.

    # TODO(thunderhoser): Need to fuck with this for dense net.  If shuffling
    # profiles together, will need 3-D predictor matrix that is reshaped to 2-D
    # in prediction function.

    # If shuffling profiles together for dense net, need to reshape
    # predictor_matrix here!
    prediction_function = _make_prediction_function(
        model_object=model_object,
        net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY]
    )

    # Find original cost (before permutation).
    print('Finding original cost (before permutation)...')
    orig_cost_estimates = _bootstrap_cost(
        target_matrices=target_matrices,
        prediction_matrices=prediction_function(predictor_matrix),
        cost_function=cost_function, num_replicates=num_bootstrap_reps
    )

    # Do dirty work.
    permuted_flag_matrix = numpy.full(
        predictor_matrix.shape[1:], False, dtype=bool
    )

    best_predictor_names = []
    best_cost_matrix = numpy.full((0, num_bootstrap_reps), numpy.nan)
    if shuffle_profiles_together:
        best_heights_m_agl = None
    else:
        best_heights_m_agl = []

    step1_predictor_names = None
    step1_heights_m_agl = None
    step1_cost_matrix = None

    step_num = 0

    while True:
        print(MINOR_SEPARATOR_STRING)
        step_num += 1

        this_result_dict = run_forward_test_one_step(
            predictor_matrix=predictor_matrix,
            target_matrices=target_matrices,
            prediction_function=prediction_function,
            cost_function=cost_function,
            permuted_flag_matrix=permuted_flag_matrix,
            shuffle_profiles_together=shuffle_profiles_together,
            num_bootstrap_reps=num_bootstrap_reps
        )

        if this_result_dict is None:
            break

        predictor_matrix = this_result_dict[PREDICTORS_KEY]
        permuted_flag_matrix = this_result_dict[PERMUTED_FLAGS_KEY]

        these_predictor_names, these_heights_m_agl = (
            _predictor_indices_to_metadata(
                all_predictor_name_matrix=predictor_name_matrix,
                all_height_matrix_m_agl=height_matrix_m_agl,
                one_step_result_dict=this_result_dict
            )
        )

        this_best_index = numpy.argmax(
            numpy.mean(this_result_dict[PERMUTED_COSTS_KEY], axis=1)
        )
        best_predictor_names.append(these_predictor_names[this_best_index])
        best_cost_matrix = numpy.concatenate((
            best_cost_matrix,
            this_result_dict[PERMUTED_COSTS_KEY][[this_best_index], :]
        ), axis=0)

        if not shuffle_profiles_together:
            best_heights_m_agl.append(these_heights_m_agl[this_best_index])

        if step_num != 1:
            continue

        step1_predictor_names = copy.deepcopy(these_predictor_names)
        step1_cost_matrix = this_result_dict[PERMUTED_COSTS_KEY] + 0.
        if not shuffle_profiles_together:
            step1_heights_m_agl = these_heights_m_agl + 0

    return {
        ORIGINAL_COST_KEY: orig_cost_estimates,
        BEST_PREDICTORS_KEY: best_predictor_names,
        BEST_HEIGHTS_KEY: best_heights_m_agl,
        BEST_COSTS_KEY: best_cost_matrix,
        STEP1_PREDICTORS_KEY: step1_predictor_names,
        STEP1_HEIGHTS_KEY: step1_heights_m_agl,
        STEP1_COSTS_KEY: step1_cost_matrix
    }
