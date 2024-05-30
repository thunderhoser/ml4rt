"""Methods for normalizing predictor and target variables."""

import numpy
import scipy.stats
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_MICRONS = 1e6

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6

VECTOR_PREDICTOR_DIM = 'vector_predictor_name'
VECTOR_TARGET_DIM = 'vector_target_name'
SCALAR_PREDICTOR_DIM = 'scalar_predictor_name'
SCALAR_TARGET_DIM = 'scalar_target_name'
WAVELENGTH_DIM = 'wavelength_metres'
HEIGHT_DIM = 'height_m_agl'
QUANTILE_LEVEL_DIM = 'quantile_level'

VECTOR_PREDICTOR_MEAN_KEY = 'vector_predictor_mean'
VECTOR_PREDICTOR_MEAN_ABS_KEY = 'vector_predictor_mean_absolute_value'
VECTOR_PREDICTOR_STDEV_KEY = 'vector_predictor_stdev'
VECTOR_PREDICTOR_QUANTILE_KEY = 'vector_predictor_quantile'
VECTOR_TARGET_MEAN_KEY = 'vector_target_mean'
VECTOR_TARGET_MEAN_ABS_KEY = 'vector_target_mean_absolute_value'
VECTOR_TARGET_STDEV_KEY = 'vector_target_stdev'
VECTOR_TARGET_QUANTILE_KEY = 'vector_target_quantile'
SCALAR_PREDICTOR_MEAN_KEY = 'scalar_predictor_mean'
SCALAR_PREDICTOR_MEAN_ABS_KEY = 'scalar_predictor_mean_absolute_value'
SCALAR_PREDICTOR_STDEV_KEY = 'scalar_predictor_stdev'
SCALAR_PREDICTOR_QUANTILE_KEY = 'scalar_predictor_quantile'
SCALAR_TARGET_MEAN_KEY = 'scalar_target_mean'
SCALAR_TARGET_MEAN_ABS_KEY = 'scalar_target_mean_absolute_value'
SCALAR_TARGET_STDEV_KEY = 'scalar_target_stdev'
SCALAR_TARGET_QUANTILE_KEY = 'scalar_target_quantile'


def _z_normalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in z-scores now.
    """

    if numpy.isnan(reference_stdev):
        data_values[:] = 0.
    else:
        data_values = (data_values - reference_mean) / reference_stdev

    return data_values


def _z_denormalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in physical units now.
    """

    if numpy.isnan(reference_stdev):
        data_values[:] = reference_mean
    else:
        data_values = reference_mean + reference_stdev * data_values

    return data_values


def _quantile_normalize_1var(data_values, reference_values_1d):
    """Does quantile normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in z-scores now.
    """

    if numpy.all(numpy.isnan(reference_values_1d)):
        data_values[numpy.isfinite(data_values)] = 0.
        return data_values

    num_quantiles = len(reference_values_1d)
    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)
    _, unique_indices = numpy.unique(reference_values_1d, return_index=True)

    if len(unique_indices) == 1:
        data_values[:] = 0.5
    else:
        interp_object = interp1d(
            x=reference_values_1d[unique_indices],
            y=quantile_levels[unique_indices],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
            assume_sorted=True
        )
        data_values = interp_object(data_values)

    # real_reference_values_1d = reference_values_1d[
    #     numpy.invert(numpy.isnan(reference_values_1d))
    # ]
    #
    # search_indices = numpy.searchsorted(
    #     a=numpy.sort(real_reference_values_1d), v=data_values, side='left'
    # ).astype(float)
    #
    # search_indices[numpy.invert(numpy.isfinite(data_values))] = numpy.nan
    # num_reference_vals = len(real_reference_values_1d)
    # data_values = search_indices / (num_reference_vals - 1)

    data_values = numpy.minimum(data_values, MAX_CUMULATIVE_DENSITY)
    data_values = numpy.maximum(data_values, MIN_CUMULATIVE_DENSITY)
    return scipy.stats.norm.ppf(data_values, loc=0., scale=1.)


def _quantile_denormalize_1var(data_values, reference_values_1d):
    """Does quantile *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in physical units now.
    """

    if numpy.all(numpy.isnan(reference_values_1d)):
        return data_values

    data_values = scipy.stats.norm.cdf(data_values, loc=0., scale=1.)
    real_reference_values_1d = reference_values_1d[
        numpy.invert(numpy.isnan(reference_values_1d))
    ]

    # Linear produces biased estimates (range of 0...0.1 in my test), while
    # lower produces unbiased estimates (range of -0.1...+0.1 in my test).
    real_flags = numpy.isfinite(data_values)
    data_values[real_flags] = numpy.percentile(
        numpy.ravel(real_reference_values_1d),
        100 * data_values[real_flags],
        interpolation='linear'
        # interpolation='lower'
    )

    return data_values


def get_normalization_params(example_dict, num_quantiles):
    """Computes normalization params for each atomic variable.

    One "atomic variable" = one field at one height (if applicable)
                            at one wavelength (if applicable)

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param num_quantiles: Number of quantiles to store for each atomic variable.
        The quantile levels will be evenly spaced from 0 to 1 (i.e., the 0th to
        100th percentile).
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    error_checking.assert_is_geq(num_quantiles, 100)
    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    num_heights = len(example_dict[example_utils.HEIGHTS_KEY])
    num_target_wavelengths = len(
        example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )
    num_vector_predictors = len(
        example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    num_vector_targets = len(
        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )
    num_scalar_predictors = len(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )
    num_scalar_targets = len(
        example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    )

    vector_predictor_mean_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    vector_predictor_mean_abs_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    vector_predictor_stdev_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    vector_predictor_quantile_matrix = numpy.full(
        (num_heights, num_vector_predictors, num_quantiles), numpy.nan
    )

    vector_target_mean_matrix = numpy.full(
        (num_heights, num_target_wavelengths, num_vector_targets), numpy.nan
    )
    vector_target_mean_abs_matrix = numpy.full(
        (num_heights, num_target_wavelengths, num_vector_targets), numpy.nan
    )
    vector_target_stdev_matrix = numpy.full(
        (num_heights, num_target_wavelengths, num_vector_targets), numpy.nan
    )
    vector_target_quantile_matrix = numpy.full(
        (num_heights, num_target_wavelengths, num_vector_targets, num_quantiles),
        numpy.nan
    )

    scalar_predictor_means = numpy.full(num_scalar_predictors, numpy.nan)
    scalar_predictor_mean_abs_values = numpy.full(
        num_scalar_predictors, numpy.nan
    )
    scalar_predictor_stdevs = numpy.full(num_scalar_predictors, numpy.nan)
    scalar_predictor_quantile_matrix = numpy.full(
        (num_scalar_predictors, num_quantiles), numpy.nan
    )

    scalar_target_mean_matrix = numpy.full(
        (num_target_wavelengths, num_scalar_targets), numpy.nan
    )
    scalar_target_mean_abs_matrix = numpy.full(
        (num_target_wavelengths, num_scalar_targets), numpy.nan
    )
    scalar_target_stdev_matrix = numpy.full(
        (num_target_wavelengths, num_scalar_targets), numpy.nan
    )
    scalar_target_quantile_matrix = numpy.full(
        (num_target_wavelengths, num_scalar_targets, num_quantiles), numpy.nan
    )

    for j in range(num_vector_predictors):
        for h in range(num_heights):
            data_values = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=
                example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY][j],
                height_m_agl=example_dict[example_utils.HEIGHTS_KEY][h]
            )

            vector_predictor_mean_matrix[h, j] = numpy.mean(data_values)
            vector_predictor_mean_abs_matrix[h, j] = numpy.mean(
                numpy.absolute(data_values)
            )
            vector_predictor_stdev_matrix[h, j] = numpy.std(
                data_values, ddof=1
            )
            vector_predictor_quantile_matrix[h, j, :] = numpy.percentile(
                data_values, 100 * quantile_levels
            )

            print((
                'Mean / mean absolute / stdev for {0:s} at {1:.0f} m AGL = '
                '{2:.4g}, {3:.4g}, {4:.4g}'
            ).format(
                example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY][j],
                example_dict[example_utils.HEIGHTS_KEY][h],
                vector_predictor_mean_matrix[h, j],
                vector_predictor_mean_abs_matrix[h, j],
                vector_predictor_stdev_matrix[h, j]
            ))

            for q in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s} at {2:.0f} m AGL = {3:.4g}'
                ).format(
                    100 * quantile_levels[q],
                    example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY][j],
                    example_dict[example_utils.HEIGHTS_KEY][h],
                    vector_predictor_quantile_matrix[h, j, q]
                ))

            print('\n')

        print(SEPARATOR_STRING)

    for j in range(num_vector_targets):
        for w in range(num_target_wavelengths):
            for h in range(num_heights):
                data_values = example_utils.get_field_from_dict(
                    example_dict=example_dict,
                    field_name=
                    example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                    height_m_agl=example_dict[example_utils.HEIGHTS_KEY][h],
                    target_wavelength_metres=
                    example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w]
                )

                vector_target_mean_matrix[h, w, j] = numpy.mean(data_values)
                vector_target_mean_abs_matrix[h, w, j] = numpy.mean(
                    numpy.absolute(data_values)
                )
                vector_target_stdev_matrix[h, w, j] = numpy.std(
                    data_values, ddof=1
                )
                vector_target_quantile_matrix[h, w, j, :] = numpy.percentile(
                    data_values, 100 * quantile_levels
                )

                print((
                    'Mean / mean absolute / stdev for {0:s} at {1:.0f} m AGL '
                    'and {2:.2f} microns = {3:.4g}, {4:.4g}, {4:.4g}'
                ).format(
                    example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                    example_dict[example_utils.HEIGHTS_KEY][h],
                    METRES_TO_MICRONS *
                    example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                    vector_target_mean_matrix[h, w, j],
                    vector_target_mean_abs_matrix[h, w, j],
                    vector_target_stdev_matrix[h, w, j]
                ))

                for q in range(num_quantiles)[::10]:
                    print((
                        '{0:.2f}th percentile for {1:s} at {2:.0f} m AGL and '
                        '{3:.2f} microns = {4:.4g}'
                    ).format(
                        100 * quantile_levels[q],
                        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                        example_dict[example_utils.HEIGHTS_KEY][h],
                        METRES_TO_MICRONS *
                        example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                        vector_target_quantile_matrix[h, w, j, q]
                    ))

                    print('\n')

                print(SEPARATOR_STRING)

    for j in range(num_scalar_predictors):
        data_values = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j]
        )

        scalar_predictor_means[j] = numpy.mean(data_values)
        scalar_predictor_mean_abs_values[j] = numpy.mean(
            numpy.absolute(data_values)
        )
        scalar_predictor_stdevs[j] = numpy.std(data_values, ddof=1)
        scalar_predictor_quantile_matrix[j, :] = numpy.percentile(
            data_values, 100 * quantile_levels
        )

        print((
            'Mean / mean absolute / stdev for {0:s} = {1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
            scalar_predictor_means[j],
            scalar_predictor_mean_abs_values[j],
            scalar_predictor_stdevs[j]
        ))

        for q in range(num_quantiles)[::10]:
            print((
                '{0:.2f}th percentile for {1:s} = {2:.4g}'
            ).format(
                100 * quantile_levels[q],
                example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
                scalar_predictor_quantile_matrix[j, q]
            ))

        print('\n')

    print(SEPARATOR_STRING)

    for j in range(num_scalar_targets):
        for w in range(num_target_wavelengths):
            data_values = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=
                example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j],
                target_wavelength_metres=
                example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w]
            )

            scalar_target_mean_matrix[w, j] = numpy.mean(data_values)
            scalar_target_mean_abs_matrix[w, j] = numpy.mean(
                numpy.absolute(data_values)
            )
            scalar_target_stdev_matrix[w, j] = numpy.std(data_values, ddof=1)
            scalar_target_quantile_matrix[w, j, :] = numpy.percentile(
                data_values, 100 * quantile_levels
            )

            print((
                'Mean / mean absolute / stdev for {0:s} at {1:.2f} microns = '
                '{2:.4g}, {3:.4g}, {4:.4g}'
            ).format(
                example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j],
                METRES_TO_MICRONS *
                example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                scalar_target_mean_matrix[w, j],
                scalar_target_mean_abs_matrix[w, j],
                scalar_target_stdev_matrix[w, j]
            ))

            for q in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s} at {2:.2f} microns = '
                    '{3:.4g}'
                ).format(
                    100 * quantile_levels[q],
                    example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j],
                    METRES_TO_MICRONS *
                    example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                    scalar_target_quantile_matrix[w, j, q]
                ))

                print('\n')

            print(SEPARATOR_STRING)

    coord_dict = {
        VECTOR_PREDICTOR_DIM:
            example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY],
        VECTOR_TARGET_DIM: example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        SCALAR_PREDICTOR_DIM:
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY],
        SCALAR_TARGET_DIM: example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        WAVELENGTH_DIM: example_dict[example_utils.TARGET_WAVELENGTHS_KEY],
        HEIGHT_DIM: example_dict[example_utils.HEIGHTS_KEY],
        QUANTILE_LEVEL_DIM: quantile_levels
    }

    main_data_dict = {
        VECTOR_PREDICTOR_MEAN_KEY: (
            (HEIGHT_DIM, VECTOR_PREDICTOR_DIM),
            vector_predictor_mean_matrix
        ),
        VECTOR_PREDICTOR_MEAN_ABS_KEY: (
            (HEIGHT_DIM, VECTOR_PREDICTOR_DIM),
            vector_predictor_mean_abs_matrix
        ),
        VECTOR_PREDICTOR_STDEV_KEY: (
            (HEIGHT_DIM, VECTOR_PREDICTOR_DIM),
            vector_predictor_stdev_matrix
        ),
        VECTOR_PREDICTOR_QUANTILE_KEY: (
            (HEIGHT_DIM, VECTOR_PREDICTOR_DIM, QUANTILE_LEVEL_DIM),
            vector_predictor_quantile_matrix
        ),
        VECTOR_TARGET_MEAN_KEY: (
            (HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_TARGET_DIM),
            vector_target_mean_matrix
        ),
        VECTOR_TARGET_MEAN_ABS_KEY: (
            (HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_TARGET_DIM),
            vector_target_mean_abs_matrix
        ),
        VECTOR_TARGET_STDEV_KEY: (
            (HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_TARGET_DIM),
            vector_target_stdev_matrix
        ),
        VECTOR_TARGET_QUANTILE_KEY: (
            (HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_TARGET_DIM, QUANTILE_LEVEL_DIM),
            vector_target_quantile_matrix
        ),
        SCALAR_PREDICTOR_MEAN_KEY: (
            (SCALAR_PREDICTOR_DIM,),
            scalar_predictor_means
        ),
        SCALAR_PREDICTOR_MEAN_ABS_KEY: (
            (SCALAR_PREDICTOR_DIM,),
            scalar_predictor_mean_abs_values
        ),
        SCALAR_PREDICTOR_STDEV_KEY: (
            (SCALAR_PREDICTOR_DIM,),
            scalar_predictor_stdevs
        ),
        SCALAR_PREDICTOR_QUANTILE_KEY: (
            (SCALAR_PREDICTOR_DIM, QUANTILE_LEVEL_DIM),
            scalar_predictor_quantile_matrix
        ),
        SCALAR_TARGET_MEAN_KEY: (
            (WAVELENGTH_DIM, SCALAR_TARGET_DIM),
            scalar_target_mean_matrix
        ),
        SCALAR_TARGET_MEAN_ABS_KEY: (
            (WAVELENGTH_DIM, SCALAR_TARGET_DIM),
            scalar_target_mean_abs_matrix
        ),
        SCALAR_TARGET_STDEV_KEY: (
            (WAVELENGTH_DIM, SCALAR_TARGET_DIM),
            scalar_target_stdev_matrix
        ),
        SCALAR_TARGET_QUANTILE_KEY: (
            (WAVELENGTH_DIM, SCALAR_TARGET_DIM, QUANTILE_LEVEL_DIM),
            scalar_target_quantile_matrix
        ),
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_data(
        example_dict, normalization_param_table_xarray, apply_to_predictors,
        apply_to_vector_targets, apply_to_scalar_targets):
    """Normalizes data.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`, containing data to be normalized.
    :param normalization_param_table_xarray: xarray table with normalization
        parameters, in format created by `get_normalization_params`.
    :param apply_to_predictors: Boolean flag.  If True, will normalize
        predictors.
    :param apply_to_vector_targets: Boolean flag.  If True, will normalize
        vector target variables.
    :param apply_to_scalar_targets: Boolean flag.  If True, will normalize
        scalar target variables.
    :return: example_dict: Same as input but with normalized values.
    :raises: ValueError: if `apply_to_predictors == apply_to_vector_targets ==
        apply_to_scalar_targets == False`.
    """

    # Check input args.
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_vector_targets)
    error_checking.assert_is_boolean(apply_to_scalar_targets)

    apply_to_any = (
        apply_to_predictors
        or apply_to_vector_targets
        or apply_to_scalar_targets
    )

    if not apply_to_any:
        raise ValueError(
            'One of `apply_to_predictors`, `apply_to_vector_targets`, and '
            '`apply_to_scalar_targets` must be True.'
        )

    # Do actual stuff.
    if apply_to_predictors:
        scalar_predictor_names = (
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
        )
    else:
        scalar_predictor_names = []

    scalar_predictor_matrix = (
        example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )
    npt = normalization_param_table_xarray

    for j in range(len(scalar_predictor_names)):
        j_new = numpy.where(
            npt.coords[SCALAR_PREDICTOR_DIM].values == scalar_predictor_names[j]
        )[0][0]

        scalar_predictor_matrix[:, j] = _quantile_normalize_1var(
            data_values=scalar_predictor_matrix[:, j],
            reference_values_1d=
            npt[SCALAR_PREDICTOR_QUANTILE_KEY].values[j_new, :]
        )

    example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        scalar_predictor_matrix
    )

    if apply_to_predictors:
        vector_predictor_names = (
            example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        )
    else:
        vector_predictor_names = []

    vector_predictor_matrix = (
        example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]

    for j in range(len(vector_predictor_names)):
        j_new = numpy.where(
            npt.coords[VECTOR_PREDICTOR_DIM].values == vector_predictor_names[j]
        )[0][0]

        for h in range(len(heights_m_agl)):
            h_new = example_utils.match_heights(
                heights_m_agl=npt.coords[HEIGHT_DIM].values,
                desired_height_m_agl=heights_m_agl[h]
            )

            vector_predictor_matrix[:, h, j] = _quantile_normalize_1var(
                data_values=vector_predictor_matrix[:, h, j],
                reference_values_1d=
                npt[VECTOR_PREDICTOR_QUANTILE_KEY].values[h_new, j_new, :]
            )

    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        vector_predictor_matrix
    )

    if apply_to_scalar_targets:
        scalar_target_names = (
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
        )
    else:
        scalar_target_names = []

    scalar_target_matrix = example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    target_wavelengths_metres = (
        example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )

    for j in range(len(scalar_target_names)):
        j_new = numpy.where(
            npt.coords[SCALAR_TARGET_DIM].values == scalar_target_names[j]
        )[0][0]

        for w in range(len(target_wavelengths_metres)):
            w_new = example_utils.match_wavelengths(
                wavelengths_metres=npt.coords[WAVELENGTH_DIM].values,
                desired_wavelength_metres=target_wavelengths_metres[w]
            )

            scalar_target_matrix[:, w, j] = _quantile_normalize_1var(
                data_values=scalar_target_matrix[:, w, j],
                reference_values_1d=
                npt[SCALAR_TARGET_QUANTILE_KEY].values[w_new, j_new, :]
            )

    example_dict[example_utils.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    if apply_to_vector_targets:
        vector_target_names = (
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
        )
    else:
        vector_target_names = []

    vector_target_matrix = example_dict[example_utils.VECTOR_TARGET_VALS_KEY]

    for j in range(len(vector_target_names)):
        j_new = numpy.where(
            npt.coords[VECTOR_TARGET_DIM].values == vector_target_names[j]
        )[0][0]

        for w in range(len(target_wavelengths_metres)):
            w_new = example_utils.match_wavelengths(
                wavelengths_metres=npt.coords[WAVELENGTH_DIM].values,
                desired_wavelength_metres=target_wavelengths_metres[w]
            )

            for h in range(len(heights_m_agl)):
                h_new = example_utils.match_heights(
                    heights_m_agl=npt.coords[HEIGHT_DIM].values,
                    desired_height_m_agl=heights_m_agl[h]
                )

                vector_target_matrix[:, h, w, j] = _quantile_normalize_1var(
                    data_values=vector_target_matrix[:, h, w, j],
                    reference_values_1d=
                    npt[VECTOR_TARGET_QUANTILE_KEY].values[h_new, w_new, j_new, :]
                )

    example_dict[example_utils.VECTOR_TARGET_VALS_KEY] = vector_target_matrix
    return example_dict


def denormalize_data(
        example_dict, normalization_param_table_xarray, apply_to_predictors,
        apply_to_vector_targets, apply_to_scalar_targets):
    """Denormalizes data.

    This method is the inverse of `normalize_data`.

    :param example_dict: See doc for `normalize_data`.
    :param normalization_param_table_xarray: Same.
    :param apply_to_predictors: Same.
    :param apply_to_vector_targets: Same.
    :param apply_to_scalar_targets: Same.
    :return: example_dict: Same.
    :raises: ValueError: if `apply_to_predictors == apply_to_vector_targets ==
        apply_to_scalar_targets == False`.
    """

    # Check input args.
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_vector_targets)
    error_checking.assert_is_boolean(apply_to_scalar_targets)

    apply_to_any = (
        apply_to_predictors
        or apply_to_vector_targets
        or apply_to_scalar_targets
    )

    if not apply_to_any:
        raise ValueError(
            'One of `apply_to_predictors`, `apply_to_vector_targets`, and '
            '`apply_to_scalar_targets` must be True.'
        )

    # Do actual stuff.
    if apply_to_predictors:
        scalar_predictor_names = (
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
        )
    else:
        scalar_predictor_names = []

    scalar_predictor_matrix = (
        example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )
    npt = normalization_param_table_xarray

    for j in range(len(scalar_predictor_names)):
        j_new = numpy.where(
            npt.coords[SCALAR_PREDICTOR_DIM].values == scalar_predictor_names[j]
        )[0][0]

        scalar_predictor_matrix[:, j] = _quantile_denormalize_1var(
            data_values=scalar_predictor_matrix[:, j],
            reference_values_1d=
            npt[SCALAR_PREDICTOR_QUANTILE_KEY].values[j_new, :]
        )

    example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        scalar_predictor_matrix
    )

    if apply_to_predictors:
        vector_predictor_names = (
            example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        )
    else:
        vector_predictor_names = []

    vector_predictor_matrix = (
        example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]

    for j in range(len(vector_predictor_names)):
        j_new = numpy.where(
            npt.coords[VECTOR_PREDICTOR_DIM].values == vector_predictor_names[j]
        )[0][0]

        for h in range(len(heights_m_agl)):
            h_new = example_utils.match_heights(
                heights_m_agl=npt.coords[HEIGHT_DIM].values,
                desired_height_m_agl=heights_m_agl[h]
            )

            vector_predictor_matrix[:, h, j] = _quantile_denormalize_1var(
                data_values=vector_predictor_matrix[:, h, j],
                reference_values_1d=
                npt[VECTOR_PREDICTOR_QUANTILE_KEY].values[h_new, j_new, :]
            )

    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        vector_predictor_matrix
    )

    if apply_to_scalar_targets:
        scalar_target_names = (
            example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
        )
    else:
        scalar_target_names = []

    scalar_target_matrix = example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    target_wavelengths_metres = (
        example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )

    for j in range(len(scalar_target_names)):
        j_new = numpy.where(
            npt.coords[SCALAR_TARGET_DIM].values == scalar_target_names[j]
        )[0][0]

        for w in range(len(target_wavelengths_metres)):
            w_new = example_utils.match_wavelengths(
                wavelengths_metres=npt.coords[WAVELENGTH_DIM].values,
                desired_wavelength_metres=target_wavelengths_metres[w]
            )

            scalar_target_matrix[:, w, j] = _quantile_denormalize_1var(
                data_values=scalar_target_matrix[:, w, j],
                reference_values_1d=
                npt[SCALAR_TARGET_QUANTILE_KEY].values[w_new, j_new, :]
            )

    example_dict[example_utils.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    if apply_to_vector_targets:
        vector_target_names = (
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
        )
    else:
        vector_target_names = []

    vector_target_matrix = example_dict[example_utils.VECTOR_TARGET_VALS_KEY]

    for j in range(len(vector_target_names)):
        j_new = numpy.where(
            npt.coords[VECTOR_TARGET_DIM].values == vector_target_names[j]
        )[0][0]

        for w in range(len(target_wavelengths_metres)):
            w_new = example_utils.match_wavelengths(
                wavelengths_metres=npt.coords[WAVELENGTH_DIM].values,
                desired_wavelength_metres=target_wavelengths_metres[w]
            )

            for h in range(len(heights_m_agl)):
                h_new = example_utils.match_heights(
                    heights_m_agl=npt.coords[HEIGHT_DIM].values,
                    desired_height_m_agl=heights_m_agl[h]
                )

                vector_target_matrix[:, h, w, j] = _quantile_denormalize_1var(
                    data_values=vector_target_matrix[:, h, w, j],
                    reference_values_1d=
                    npt[VECTOR_TARGET_QUANTILE_KEY].values[h_new, w_new, j_new, :]
                )

    example_dict[example_utils.VECTOR_TARGET_VALS_KEY] = vector_target_matrix
    return example_dict


def create_mean_example(example_dict, normalization_param_table_xarray,
                        use_absolute_values=False):
    """Creates mean example (with mean value for each atomic variable).

    One "atomic variable" = one field at one height (if applicable)
                            at one wavelength (if applicable)

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`, containing data to be normalized.
    :param normalization_param_table_xarray: xarray table with normalization
        parameters, in format created by `get_normalization_params`.
    :param use_absolute_values: Boolean flag.  If True, will create mean
        *absolute* example.
    :return: example_dict: Same as input but containing only one "example,"
        i.e., the mean.
    """

    scalar_predictor_names = example_dict[
        example_utils.SCALAR_PREDICTOR_NAMES_KEY
    ]
    vector_predictor_names = example_dict[
        example_utils.VECTOR_PREDICTOR_NAMES_KEY
    ]
    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    target_wavelengths_metres = example_dict[
        example_utils.TARGET_WAVELENGTHS_KEY
    ]

    npt = normalization_param_table_xarray
    js = numpy.array([
        numpy.where(npt.coords[SCALAR_PREDICTOR_DIM].values == this_name)[0][0]
        for this_name in scalar_predictor_names
    ], dtype=int)

    if use_absolute_values:
        mean_scalar_predictor_values = (
            npt[SCALAR_PREDICTOR_MEAN_ABS_KEY].values[js]
        )
    else:
        mean_scalar_predictor_values = npt[SCALAR_PREDICTOR_MEAN_KEY].values[js]

    js = numpy.array([
        numpy.where(npt.coords[VECTOR_PREDICTOR_DIM].values == this_name)[0][0]
        for this_name in vector_predictor_names
    ], dtype=int)
    hs = numpy.array([
        example_utils.match_heights(
            heights_m_agl=npt.coords[HEIGHT_DIM].values,
            desired_height_m_agl=this_height
        )
        for this_height in heights_m_agl
    ], dtype=int)

    if use_absolute_values:
        mean_vector_predictor_matrix = (
            npt[VECTOR_PREDICTOR_MEAN_ABS_KEY].values[hs, :][:, js]
        )
    else:
        mean_vector_predictor_matrix = (
            npt[VECTOR_PREDICTOR_MEAN_KEY].values[hs, :][:, js]
        )

    js = numpy.array([
        numpy.where(npt.coords[SCALAR_TARGET_DIM].values == this_name)[0][0]
        for this_name in scalar_target_names
    ], dtype=int)
    ws = numpy.array([
        example_utils.match_wavelengths(
            wavelengths_metres=npt.coords[WAVELENGTH_DIM].values,
            desired_wavelength_metres=this_wavelength
        )
        for this_wavelength in target_wavelengths_metres
    ], dtype=int)

    if use_absolute_values:
        mean_scalar_target_matrix = (
            npt[SCALAR_TARGET_MEAN_ABS_KEY].values[ws, :][:, js]
        )
    else:
        mean_scalar_target_matrix = (
            npt[SCALAR_TARGET_MEAN_KEY].values[ws, :][:, js]
        )

    js = numpy.array([
        numpy.where(npt.coords[VECTOR_TARGET_DIM].values == this_name)[0][0]
        for this_name in vector_target_names
    ], dtype=int)

    if use_absolute_values:
        mean_vector_target_matrix = (
            npt[VECTOR_TARGET_MEAN_ABS_KEY].values[hs, ...][:, ws, :][..., js]
        )
    else:
        mean_vector_target_matrix = (
            npt[VECTOR_TARGET_MEAN_KEY].values[hs, ...][:, ws, :][..., js]
        )

    return {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_utils.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_utils.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.TARGET_WAVELENGTHS_KEY: target_wavelengths_metres,
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_scalar_predictor_values, axis=0),
        example_utils.SCALAR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_scalar_target_matrix, axis=0),
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_vector_predictor_matrix, axis=0),
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_vector_target_matrix, axis=0)
    }


def create_mean_example_old(new_example_dict, training_example_dict):
    """Creates mean example (with mean value for each variable/height pair).

    :param new_example_dict: See doc for `normalize_data`.
    :param training_example_dict: Same.
    :return: mean_example_dict: See doc for `example_utils.average_examples`.
    """

    scalar_predictor_names = new_example_dict[
        example_utils.SCALAR_PREDICTOR_NAMES_KEY
    ]
    scalar_target_names = new_example_dict[
        example_utils.SCALAR_TARGET_NAMES_KEY
    ]
    vector_predictor_names = new_example_dict[
        example_utils.VECTOR_PREDICTOR_NAMES_KEY
    ]
    vector_target_names = new_example_dict[
        example_utils.VECTOR_TARGET_NAMES_KEY
    ]
    heights_m_agl = new_example_dict[
        example_utils.HEIGHTS_KEY
    ]
    target_wavelengths_metres = new_example_dict[
        example_utils.TARGET_WAVELENGTHS_KEY
    ]

    num_scalar_predictors = len(scalar_predictor_names)
    num_scalar_targets = len(scalar_target_names)
    num_vector_predictors = len(vector_predictor_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)
    num_target_wavelengths = len(target_wavelengths_metres)

    mean_scalar_predictor_values = numpy.full(num_scalar_predictors, numpy.nan)
    mean_scalar_target_matrix = numpy.full(
        (num_target_wavelengths, num_scalar_targets), numpy.nan
    )
    mean_vector_predictor_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    mean_vector_target_matrix = numpy.full(
        (num_heights, num_target_wavelengths, num_vector_targets), numpy.nan
    )

    for k in range(num_scalar_predictors):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_predictor_names[k]
        )
        mean_scalar_predictor_values[k] = numpy.mean(these_training_values)

    for k in range(num_scalar_targets):
        for w in range(num_target_wavelengths):
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=scalar_target_names[k],
                target_wavelength_metres=target_wavelengths_metres[w]
            )
            mean_scalar_target_matrix[w, k] = numpy.mean(these_training_values)

    for j in range(num_heights):
        for k in range(num_vector_predictors):
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_predictor_names[k],
                height_m_agl=heights_m_agl[j]
            )
            mean_vector_predictor_matrix[j, k] = numpy.mean(
                these_training_values
            )

        for k in range(num_vector_targets):
            for w in range(num_target_wavelengths):
                these_training_values = example_utils.get_field_from_dict(
                    example_dict=training_example_dict,
                    field_name=vector_target_names[k],
                    height_m_agl=heights_m_agl[j],
                    target_wavelength_metres=target_wavelengths_metres[w]
                )
                mean_vector_target_matrix[j, w, k] = numpy.mean(
                    these_training_values
                )

    return {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_utils.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_utils.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.TARGET_WAVELENGTHS_KEY: target_wavelengths_metres,
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_scalar_predictor_values, axis=0),
        example_utils.SCALAR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_scalar_target_matrix, axis=0),
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_vector_predictor_matrix, axis=0),
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_vector_target_matrix, axis=0)
    }


def write_params(normalization_param_table_xarray, netcdf_file_name):
    """Writes normalization parameters to NetCDF file.

    :param normalization_param_table_xarray: xarray table with normalization
        parameters, in format created by `get_normalization_params`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    normalization_param_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_params(netcdf_file_name):
    """Reads normalization parameters for from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters, in format created by `get_normalization_params`.
    """

    error_checking.assert_file_exists(netcdf_file_name)
    return xarray.open_dataset(netcdf_file_name)
