"""Piecewise-linear approximation to exact normalization method."""

import numpy
import xarray
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import example_utils

SCALAR_PREDICTOR_DIM = 'scalar_predictor'
VECTOR_PREDICTOR_DIM = 'vector_predictor'
HEIGHT_DIM = 'height'
LINEAR_PIECE_DIM = 'linear_piece'
BREAK_POINT_DIM = 'break_point'

SCALAR_BREAK_POINT_KEY = 'scalar_break_point_physical_units'
SCALAR_SLOPE_KEY = 'scalar_slope'
SCALAR_INTERCEPT_KEY = 'scalar_intercept'
VECTOR_BREAK_POINT_KEY = 'vector_break_point_physical_units'
VECTOR_SLOPE_KEY = 'vector_slope'
VECTOR_INTERCEPT_KEY = 'vector_intercept'


def _fit_piecewise_linear_model_for_unif_1var(
        physical_values, uniformized_values, num_linear_pieces,
        max_acceptable_error):
    """Fits piecewise-linear model for uniformization.

    This method handles one variable -- either one scalar variable or one vector
    variable at one height.

    V = number of reference values
    P = number of linear pieces in model

    :param physical_values: length-V numpy array of physical values.
    :param uniformized_values: length-V numpy array of corresponding uniformized
        values.
    :param num_linear_pieces: P in the above discussion.
    :param max_acceptable_error: Max acceptable error.
    :return: model_break_points_physical_units: length-(P + 1) numpy array of
        break points for the line segments.
    :return: model_slopes: length-P numpy array of slopes.
    :return: model_intercepts: length-P numpy array of intercepts.
    :raises: ValueError: if the max acceptable error is exceeded by any example.
    """

    first_guess_break_points_unif = numpy.linspace(
        0, 1, num=num_linear_pieces + 1, dtype=float
    )
    break_point_indices = numpy.array([
        numpy.argmin(numpy.absolute(uniformized_values - b))
        for b in first_guess_break_points_unif
    ], dtype=int)

    model_break_points_phys = physical_values[break_point_indices]
    model_break_points_unif = uniformized_values[break_point_indices]

    model_break_points_phys, unique_indices = numpy.unique(
        model_break_points_phys, return_index=True
    )
    model_break_points_unif = model_break_points_unif[unique_indices]

    if len(model_break_points_phys) == 1:
        return (
            model_break_points_phys,
            numpy.array([], dtype=float),
            numpy.array([], dtype=float)
        )

    model_slopes = (
        numpy.diff(model_break_points_unif) /
        numpy.diff(model_break_points_phys)
    )
    model_slopes[numpy.isnan(model_slopes)] = 0.
    model_intercepts = (
        model_break_points_unif[:-1] -
        model_slopes * model_break_points_phys[:-1]
    )

    estimated_uniformized_values = _apply_piecewise_linear_model_for_unif_1var(
        input_values_physical_units=physical_values,
        model_break_points_physical_units=model_break_points_phys,
        model_slopes=model_slopes,
        model_intercepts=model_intercepts
    )

    absolute_errors = numpy.absolute(
        uniformized_values - estimated_uniformized_values
    )
    print('Max absolute error = {0:.4f}'.format(
        numpy.max(absolute_errors)
    ))

    if numpy.any(absolute_errors > max_acceptable_error):
        error_string = (
            'POTENTIAL ERROR: {0:d} of {1:d} predictions have an absolute '
            'error above {2:.4f}.  Absolute errors are sorted in descending '
            'order below:\n{3:s}'
        ).format(
            numpy.sum(absolute_errors > max_acceptable_error),
            len(absolute_errors),
            max_acceptable_error,
            str(numpy.sort(absolute_errors)[::-1])
        )

        # warnings.warn(error_string)
        raise ValueError(error_string)

    return model_break_points_phys, model_slopes, model_intercepts


def _apply_piecewise_linear_model_for_unif_1var(
        input_values_physical_units, model_break_points_physical_units,
        model_slopes, model_intercepts):
    """Applies piecewise-linear model for uniformization.

    This method handles one variable -- either one scalar variable or one vector
    variable at one height.

    V = number of input values to transform
    P = number of linear pieces in model

    :param input_values_physical_units: length-V numpy array of input values to
        transform.
    :param model_break_points_physical_units: length-(P + 1) numpy array of
        break points for the line segments.
    :param model_slopes: length-P numpy array of slopes.
    :param model_intercepts: length-P numpy array of intercepts.
    :return: output_values_uniformized: length-V numpy array of transformed
        output values.
    """

    num_linear_pieces = len(model_slopes)
    num_examples = len(input_values_physical_units)
    output_values_uniformized = numpy.full(num_examples, numpy.nan)

    if num_linear_pieces == 0:
        output_values_uniformized[:] = 0.
        return output_values_uniformized

    for i in range(num_linear_pieces):
        if i == num_linear_pieces - 1:
            these_flags = (
                input_values_physical_units >=
                model_break_points_physical_units[i]
            )
        elif i == 0:
            these_flags = (
                input_values_physical_units <
                model_break_points_physical_units[i + 1]
            )
        else:
            these_flags = numpy.logical_and(
                input_values_physical_units >=
                model_break_points_physical_units[i],
                input_values_physical_units <
                model_break_points_physical_units[i + 1]
            )

        these_indices = numpy.where(numpy.logical_and(
            these_flags,
            numpy.isnan(output_values_uniformized)
        ))[0]

        if len(these_indices) == 0:
            continue

        output_values_uniformized[these_indices] = (
            model_slopes[i] * input_values_physical_units[these_indices]
            + model_intercepts[i]
        )

    assert not numpy.any(numpy.isnan(output_values_uniformized))
    return output_values_uniformized


def fit_piecewise_linear_models_for_unif(
        training_example_dict, num_linear_pieces, max_acceptable_error):
    """For every predictor, fits pcwise-linear model to approx uniformization.

    :param training_example_dict: Dictionary with training examples (see doc for
        `example_io.read_file`).
    :param num_linear_pieces: Number of linear pieces in each model.
    :param max_acceptable_error: Max acceptable absolute error (for a single
        transformed value for any predictor).
    :return: pw_linear_model_table_xarray: xarray table.  Metadata and variable
        names in the table should make it self-explanatory.
    """

    error_checking.assert_is_integer(num_linear_pieces)
    error_checking.assert_is_geq(num_linear_pieces, 10)
    error_checking.assert_is_greater(max_acceptable_error, 0.)
    error_checking.assert_is_leq(max_acceptable_error, 0.1)

    scalar_predictor_names = (
        training_example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )

    num_scalar_predictors = len(scalar_predictor_names)
    scalar_break_point_matrix = numpy.full(
        (num_scalar_predictors, num_linear_pieces + 1), numpy.nan
    )
    scalar_slope_matrix = numpy.full(
        (num_scalar_predictors, num_linear_pieces), numpy.nan
    )
    scalar_intercept_matrix = numpy.full(
        (num_scalar_predictors, num_linear_pieces), numpy.nan
    )

    for j in range(num_scalar_predictors):
        print('Fitting piecewise-linear model to uniformize {0:s}...'.format(
            scalar_predictor_names[j]
        ))

        these_physical_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_predictor_names[j]
        )
        these_uniformized_values = _orig_to_uniform_dist(
            orig_values_new=these_physical_values + 0.,
            orig_values_training=these_physical_values
        )

        (
            these_break_points, these_slopes, these_intercepts
        ) = _fit_piecewise_linear_model_for_unif_1var(
            physical_values=these_physical_values,
            uniformized_values=these_uniformized_values,
            num_linear_pieces=num_linear_pieces,
            max_acceptable_error=max_acceptable_error
        )

        scalar_break_point_matrix[j, :len(these_break_points)] = (
            these_break_points
        )
        scalar_slope_matrix[j, :len(these_slopes)] = these_slopes
        scalar_intercept_matrix[j, :len(these_intercepts)] = these_intercepts

    vector_predictor_names = (
        training_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    heights_m_agl = training_example_dict[example_utils.HEIGHTS_KEY]

    num_vector_predictors = len(vector_predictor_names)
    num_heights = len(heights_m_agl)
    vector_break_point_matrix = numpy.full(
        (num_vector_predictors, num_heights, num_linear_pieces + 1), numpy.nan
    )
    vector_slope_matrix = numpy.full(
        (num_vector_predictors, num_heights, num_linear_pieces), numpy.nan
    )
    vector_intercept_matrix = numpy.full(
        (num_vector_predictors, num_heights, num_linear_pieces), numpy.nan
    )

    for j in range(num_vector_predictors):
        for k in range(num_heights):
            print((
                'Fitting piecewise-linear model to uniformize {0:s} at {1:d} '
                'm AGL...'
            ).format(
                vector_predictor_names[j],
                int(numpy.round(heights_m_agl[k]))
            ))

            these_physical_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_predictor_names[j],
                height_m_agl=heights_m_agl[k]
            )
            these_uniformized_values = _orig_to_uniform_dist(
                orig_values_new=these_physical_values + 0.,
                orig_values_training=these_physical_values
            )

            (
                these_break_points, these_slopes, these_intercepts
            ) = _fit_piecewise_linear_model_for_unif_1var(
                physical_values=these_physical_values,
                uniformized_values=these_uniformized_values,
                num_linear_pieces=num_linear_pieces,
                max_acceptable_error=max_acceptable_error
            )

            vector_break_point_matrix[j, k, :len(these_break_points)] = (
                these_break_points
            )
            vector_slope_matrix[j, k, :len(these_slopes)] = these_slopes
            vector_intercept_matrix[j, k, :len(these_intercepts)] = (
                these_intercepts
            )

    coord_dict = {
        SCALAR_PREDICTOR_DIM: scalar_predictor_names,
        VECTOR_PREDICTOR_DIM: vector_predictor_names,
        HEIGHT_DIM: heights_m_agl
    }

    main_data_dict = {
        SCALAR_BREAK_POINT_KEY: (
            (SCALAR_PREDICTOR_DIM, BREAK_POINT_DIM),
            scalar_break_point_matrix
        ),
        SCALAR_SLOPE_KEY: (
            (SCALAR_PREDICTOR_DIM, LINEAR_PIECE_DIM),
            scalar_slope_matrix
        ),
        SCALAR_INTERCEPT_KEY: (
            (SCALAR_PREDICTOR_DIM, LINEAR_PIECE_DIM),
            scalar_intercept_matrix
        ),
        VECTOR_BREAK_POINT_KEY: (
            (VECTOR_PREDICTOR_DIM, HEIGHT_DIM, BREAK_POINT_DIM),
            vector_break_point_matrix
        ),
        VECTOR_SLOPE_KEY: (
            (VECTOR_PREDICTOR_DIM, HEIGHT_DIM, LINEAR_PIECE_DIM),
            vector_slope_matrix
        ),
        VECTOR_INTERCEPT_KEY: (
            (VECTOR_PREDICTOR_DIM, HEIGHT_DIM, LINEAR_PIECE_DIM),
            vector_intercept_matrix
        )
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_data_with_pw_linear_models_for_unif(
        new_example_dict, pw_linear_model_table_xarray):
    """Same as normalize_data, but using pw-linear models to approx uniformizn.

    Also, this method has many fewer options than normalize_data.  Specifically,
    this method uses z-score normalization after uniformization, applied only to
    the predictor variables (not the target variables), applied independently at
    each height.

    :param new_example_dict: Dictionary with learning examples to be normalized
        (see doc for `example_io.read_file`).
    :param pw_linear_model_table_xarray: xarray table in format returned by
        `fit_piecewise_linear_models_for_unif`.
    :return: example_dict: Same as input but with normalized values.
    """

    scalar_predictor_names = (
        new_example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )
    new_scalar_predictor_matrix = (
        new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )
    pwlmt = pw_linear_model_table_xarray

    for j in range(len(scalar_predictor_names)):
        this_predictor_values = numpy.ravel(new_scalar_predictor_matrix[..., j])

        j_pw = numpy.where(
            pwlmt.coords[SCALAR_PREDICTOR_DIM].values ==
            scalar_predictor_names[j]
        )[0][0]

        this_num_linear_pieces = numpy.sum(numpy.invert(numpy.isnan(
            pwlmt[SCALAR_SLOPE_KEY].values[j_pw, :]
        )))
        this_num_breaks = this_num_linear_pieces + 1

        this_predictor_values = _apply_piecewise_linear_model_for_unif_1var(
            input_values_physical_units=this_predictor_values,
            model_break_points_physical_units=
            pwlmt[SCALAR_BREAK_POINT_KEY].values[j_pw, :this_num_breaks],
            model_slopes=
            pwlmt[SCALAR_SLOPE_KEY].values[j_pw, :this_num_linear_pieces],
            model_intercepts=
            pwlmt[SCALAR_INTERCEPT_KEY].values[j_pw, :this_num_linear_pieces]
        )

        this_predictor_values = _orig_to_normal_dist(this_predictor_values)
        new_scalar_predictor_matrix[..., j] = numpy.reshape(
            this_predictor_values,
            new_scalar_predictor_matrix[..., j].shape
        )

    new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        new_scalar_predictor_matrix
    )

    vector_predictor_names = (
        new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    heights_m_agl = new_example_dict[example_utils.HEIGHTS_KEY]
    new_vector_predictor_matrix = (
        new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )

    for j in range(len(vector_predictor_names)):
        for k in range(len(heights_m_agl)):
            this_predictor_values = numpy.ravel(
                new_vector_predictor_matrix[..., k, j]
            )

            j_pw = numpy.where(
                pwlmt.coords[VECTOR_PREDICTOR_DIM].values ==
                vector_predictor_names[j]
            )[0][0]

            k_pw = example_utils.match_heights(
                heights_m_agl=pwlmt.coords[HEIGHT_DIM].values,
                desired_height_m_agl=heights_m_agl[k]
            )

            this_num_linear_pieces = numpy.sum(numpy.invert(numpy.isnan(
                pwlmt[VECTOR_SLOPE_KEY].values[j_pw, k_pw, :]
            )))
            this_num_breaks = this_num_linear_pieces + 1

            this_predictor_values = _apply_piecewise_linear_model_for_unif_1var(
                input_values_physical_units=this_predictor_values,
                model_break_points_physical_units=
                pwlmt[VECTOR_BREAK_POINT_KEY].values[j_pw, k_pw, :this_num_breaks],
                model_slopes=
                pwlmt[VECTOR_SLOPE_KEY].values[j_pw, k_pw, :this_num_linear_pieces],
                model_intercepts=
                pwlmt[VECTOR_INTERCEPT_KEY].values[j_pw, k_pw, :this_num_linear_pieces]
            )

            this_predictor_values = _orig_to_normal_dist(this_predictor_values)
            new_vector_predictor_matrix[..., k, j] = numpy.reshape(
                this_predictor_values,
                new_vector_predictor_matrix[..., k, j].shape
            )

    new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        new_vector_predictor_matrix
    )

    return new_example_dict


def write_piecewise_linear_models_for_unif(
        pw_linear_model_table_xarray, netcdf_file_name):
    """Writes piecewise-linear models for uniformization (one per predictor).

    :param pw_linear_model_table_xarray: xarray table in format returned by
        `fit_piecewise_linear_models_for_unif`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)

    pw_linear_model_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_piecewise_linear_models_for_unif(netcdf_file_name):
    """Reads piecewise-linear models for uniformization (one per predictor).

    :param netcdf_file_name: Path to input file.
    :return: pw_linear_model_table_xarray: xarray table in format returned by
        `fit_piecewise_linear_models_for_unif`.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
