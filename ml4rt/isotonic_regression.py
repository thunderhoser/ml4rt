"""Methods for building, training, and applying isotonic-regression models."""

import os
import sys
import pickle
import numpy
from sklearn.isotonic import IsotonicRegression

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking


def train_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        vector_target_matrix, scalar_target_matrix, separate_by_height=True):
    """Trains isotonic-regression models.

    E = number of examples
    H = number of heights
    T_v = number of vector target variables
    T_s = number of scalar target variables
    S = number of ensemble members

    :param orig_vector_prediction_matrix: numpy array (E x H x T_v x S) of
        predicted values for vector target variables.
    :param orig_scalar_prediction_matrix: numpy array (E x T_s x S) of predicted
        values for scalar target variables.
    :param vector_target_matrix: numpy array (E x H x T_v) of actual values
        for vector target variables.
    :param scalar_target_matrix: numpy array (E x T_s) of actual values for
        scalar target variables.
    :param separate_by_height: Boolean flag.  If True, will train one model for
        each target variable (channel).  If False, will train one model for each
        pair of target variable and height.
    :return: scalar_model_objects: List (length T_s) of models (instances of
        `sklearn.isotonic.IsotonicRegression`) for scalar target variables.
    :return: vector_model_object_matrix: numpy array of models
        (instances of `sklearn.isotonic.IsotonicRegression`) for vector target
        variables.  If `separate_by_height == True`, this array is H x T_v.
        If `separate_by_height == False`, this array has is 1 x T_v.
    """

    # Check input args.
    num_examples = None
    num_heights = 0
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_target_matrix is not None
    )

    if have_vectors:
        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=4
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_target_matrix,
            exact_dimensions=numpy.array(
                orig_vector_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)

        num_examples = vector_target_matrix.shape[0]
        num_heights = vector_target_matrix.shape[1]
        num_vector_targets = vector_target_matrix.shape[2]
        ensemble_size = orig_vector_prediction_matrix.shape[3]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or scalar_target_matrix is not None
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=3
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[2]

        expected_dim = numpy.array([
            num_examples, orig_scalar_prediction_matrix.shape[1], ensemble_size
        ], dtype=int)

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            scalar_target_matrix,
            exact_dimensions=numpy.array(
                orig_scalar_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)

        num_scalar_targets = scalar_target_matrix.shape[1]

    error_checking.assert_is_boolean(separate_by_height)

    # Do actual stuff.
    scalar_model_objects = [None] * num_scalar_targets
    num_modeling_heights = num_heights if separate_by_height else 1
    vector_model_object_matrix = numpy.full(
        (num_modeling_heights, num_vector_targets), '', dtype=object
    )

    for k in range(num_scalar_targets):
        print((
            'Training isotonic-regression model for {0:d}th of {1:d} scalar '
            'target variables...'
        ).format(
            k + 1, num_scalar_targets
        ))

        this_prediction_matrix = orig_scalar_prediction_matrix[:, k, :] + 0.
        this_target_matrix = numpy.expand_dims(
            scalar_target_matrix[:, k], axis=-1
        )
        this_target_matrix = numpy.repeat(
            this_target_matrix, axis=-1, repeats=ensemble_size
        )

        scalar_model_objects[k] = IsotonicRegression(
            increasing=True, out_of_bounds='clip'
        )
        scalar_model_objects[k].fit(
            X=numpy.ravel(this_prediction_matrix),
            y=numpy.ravel(this_target_matrix)
        )

    if num_scalar_targets > 0:
        print('\n')

    for k in range(num_vector_targets):
        for j in range(num_modeling_heights):
            print((
                'Training isotonic-regression model for {0:d}th of {1:d} vector'
                ' target variables at {2:d}th of {3:d} modeling heights...'
            ).format(
                k + 1, num_vector_targets, j + 1, num_modeling_heights
            ))

            vector_model_object_matrix[j, k] = IsotonicRegression(
                increasing=True, out_of_bounds='clip'
            )

            if separate_by_height:
                this_prediction_matrix = (
                    orig_vector_prediction_matrix[:, j, k, :] + 0.
                )
                this_target_matrix = numpy.expand_dims(
                    vector_target_matrix[:, j, k], axis=-1
                )
                this_target_matrix = numpy.repeat(
                    this_target_matrix, axis=-1, repeats=ensemble_size
                )
            else:
                this_prediction_matrix = (
                    orig_vector_prediction_matrix[:, :, k, :] + 0.
                )
                this_target_matrix = numpy.expand_dims(
                    vector_target_matrix[:, :, k], axis=-1
                )
                this_target_matrix = numpy.repeat(
                    this_target_matrix, axis=-1, repeats=ensemble_size
                )

            vector_model_object_matrix[j, k].fit(
                X=numpy.ravel(this_prediction_matrix),
                y=numpy.ravel(this_target_matrix)
            )

        if k != num_vector_targets - 1:
            print('\n')

    return scalar_model_objects, vector_model_object_matrix


def apply_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        scalar_model_objects, vector_model_object_matrix):
    """Applies isotonic-regression models.

    :param orig_vector_prediction_matrix: See doc for `train_models`.
    :param orig_scalar_prediction_matrix: Same.
    :param scalar_model_objects: Same.
    :param vector_model_object_matrix: Same.
    :return: new_vector_prediction_matrix: Same as
        `orig_vector_prediction_matrix` but with transformed values.
    :return: new_scalar_prediction_matrix: Same as
        `orig_scalar_prediction_matrix` but with transformed values.
    """

    # Check input args.
    num_examples = None
    num_modeling_heights = 0
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_model_object_matrix.size > 0
    )

    if have_vectors:
        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=4
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, num_dimensions=2
        )

        num_modeling_heights = vector_model_object_matrix.shape[0]
        num_vector_targets = orig_vector_prediction_matrix.shape[2]
        expected_dim = numpy.array(
            [num_modeling_heights, num_vector_targets], dtype=int
        )
        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, exact_dimensions=expected_dim
        )

        num_examples = orig_vector_prediction_matrix.shape[0]
        ensemble_size = orig_vector_prediction_matrix.shape[3]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or len(scalar_model_objects) > 0
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=3
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[2]

        num_scalar_targets = orig_scalar_prediction_matrix.shape[1]
        expected_dim = numpy.array(
            [num_examples, num_scalar_targets, ensemble_size], dtype=int
        )

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            numpy.array(scalar_model_objects),
            exact_dimensions=numpy.array([num_scalar_targets], dtype=int)
        )

    if have_vectors:
        new_vector_prediction_matrix = numpy.full(
            orig_vector_prediction_matrix.shape, numpy.nan
        )
    else:
        new_vector_prediction_matrix = numpy.full(
            (num_examples, 0, 0, ensemble_size), numpy.nan
        )

    if have_scalars:
        new_scalar_prediction_matrix = numpy.full(
            orig_scalar_prediction_matrix.shape, numpy.nan
        )
    else:
        new_scalar_prediction_matrix = numpy.full(
            (num_examples, 0, ensemble_size), numpy.nan
        )

    for k in range(num_scalar_targets):
        print((
            'Applying isotonic-regression model to {0:d}th of {1:d} scalar '
            'target variables...'
        ).format(
            k + 1, num_scalar_targets
        ))

        for m in range(ensemble_size):
            new_scalar_prediction_matrix[:, k, m] = (
                scalar_model_objects[k].predict(
                    orig_scalar_prediction_matrix[:, k, m]
                )
            )

    if num_scalar_targets > 0:
        print('\n')

    for k in range(num_vector_targets):
        for j in range(num_modeling_heights):
            print((
                'Applying isotonic-regression model to {0:d}th of {1:d} vector'
                ' target variables at {2:d}th of {3:d} modeling heights...'
            ).format(
                k + 1, num_vector_targets, j + 1, num_modeling_heights
            ))

            for m in range(ensemble_size):
                if num_modeling_heights == 1:
                    these_predictions = vector_model_object_matrix[j, k].predict(
                        numpy.ravel(orig_vector_prediction_matrix[..., k, m])
                    )
                    new_vector_prediction_matrix[..., k, m] = numpy.reshape(
                        these_predictions,
                        orig_vector_prediction_matrix.shape[:2]
                    )
                else:
                    new_vector_prediction_matrix[:, j, k, m] = (
                        vector_model_object_matrix[j, k].predict(
                            orig_vector_prediction_matrix[:, j, k, m]
                        )
                    )

        if k != num_vector_targets - 1:
            print('\n')

    return new_vector_prediction_matrix, new_scalar_prediction_matrix


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with set of isotonic-regression models.

    :param model_dir_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: dill_file_name: Path to Dill file with models.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    dill_file_name = '{0:s}/isotonic_regression.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(dill_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            dill_file_name
        )
        raise ValueError(error_string)

    return dill_file_name


def write_file(dill_file_name, scalar_model_objects,
               vector_model_object_matrix):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param scalar_model_objects: See doc for `train_models`.
    :param vector_model_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(scalar_model_objects), num_dimensions=1
    )
    error_checking.assert_is_numpy_array(
        vector_model_object_matrix, num_dimensions=2
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    pickle.dump(scalar_model_objects, dill_file_handle)
    pickle.dump(vector_model_object_matrix, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: scalar_model_objects: See doc for `train_models`.
    :return: vector_model_object_matrix: Same.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    scalar_model_objects = pickle.load(dill_file_handle)
    vector_model_object_matrix = pickle.load(dill_file_handle)
    dill_file_handle.close()

    return scalar_model_objects, vector_model_object_matrix
