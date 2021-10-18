"""Methods for 1-D interpolation of heating rate over height."""

import numpy
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
LARGE_NUMBER = 1e10

DAYS_TO_SECONDS = 86400
SPECIFIC_HEAT_J_KG01_K01 = 3.5 * 287.04


def _find_local_maxima(orig_heating_rate_matrix_k_day01, orig_heights_m_agl,
                       new_heights_m_agl, half_window_size_px):
    """Finds local maxima in each heating-rate profile.

    E = number of examples
    h = number of heights in original grid
    H = number of heights in new grid

    :param orig_heating_rate_matrix_k_day01: See doc for `interpolate`.
    :param orig_heights_m_agl: Same.
    :param new_heights_m_agl: Same.
    :param half_window_size_px: Same.
    :return: max_flag_matrix: E-by-H numpy array of Boolean flags, indicating
        where there is a local maximum.
    :return: max_heating_rate_matrix_k_day01: E-by-h numpy array of max-filtered
        heating rates.
    """

    max_heating_rate_matrix_k_day01 = maximum_filter1d(
        orig_heating_rate_matrix_k_day01, size=2 * half_window_size_px + 1,
        axis=-1, mode='constant', cval=0.
    )
    max_flag_matrix = numpy.isclose(
        orig_heating_rate_matrix_k_day01, max_heating_rate_matrix_k_day01,
        atol=TOLERANCE
    )

    # new_to_orig_height_indices = numpy.array([
    #     numpy.argmin(numpy.absolute(h - orig_heights_m_agl))
    #     for h in new_heights_m_agl
    # ], dtype=int)
    #
    # return max_flag_matrix[..., new_to_orig_height_indices]

    orig_to_new_height_indices = numpy.array([
        numpy.argmin(numpy.absolute(h - new_heights_m_agl))
        for h in orig_heights_m_agl
    ], dtype=int)

    num_examples = max_flag_matrix.shape[0]
    num_orig_heights = len(orig_heights_m_agl)
    num_new_heights = len(new_heights_m_agl)
    new_max_flag_matrix = numpy.full(
        (num_examples, num_new_heights), 0, dtype=bool
    )

    for j in range(num_orig_heights):
        k = orig_to_new_height_indices[j]
        new_max_flag_matrix[:, k] = numpy.logical_or(
            new_max_flag_matrix[:, k], max_flag_matrix[:, j]
        )

    return new_max_flag_matrix, max_heating_rate_matrix_k_day01


def _conserve_heating(
        orig_heating_rate_matrix_k_day01, orig_air_dens_matrix_kg_m03,
        orig_heights_metres, new_heating_rate_matrix_k_day01,
        new_air_dens_matrix_kg_m03, new_heights_metres, test_mode=False):
    """Conserves column-total heating between two grids.

    E = number of examples
    h = number of heights in original grid
    H = number of heights in new grid

    :param orig_heating_rate_matrix_k_day01: E-by-h numpy array of heating rates
        (Kelvins per day).
    :param orig_air_dens_matrix_kg_m03: E-by-h numpy array of air densities.
    :param orig_heights_metres: length-h numpy array of heights.
    :param new_heating_rate_matrix_k_day01: E-by-H numpy array of heating rates
        (Kelvins per day).
    :param new_air_dens_matrix_kg_m03: E-by-H numpy array of air densities.
    :param new_heights_metres: length-H numpy array of heights.
    :param test_mode: Leave this alone.
    :return: new_heating_rate_matrix_k_day01: Same as input but conserving
        column-total heating.
    """

    raise ValueError(
        'This method may be too difficult to use, since it requires air-density'
        ' profiles, which requires reading predictors in addition to'
        ' predictions.'
    )

    orig_heating_rate_matrix_k_s01 = (
        orig_heating_rate_matrix_k_day01 / DAYS_TO_SECONDS
    )
    new_heating_rate_matrix_k_s01 = (
        new_heating_rate_matrix_k_day01 / DAYS_TO_SECONDS
    )

    if test_mode:
        num_examples = orig_heating_rate_matrix_k_s01.shape[0]
        orig_height_matrix_metres = numpy.expand_dims(
            orig_heights_metres, axis=0
        )
        orig_height_matrix_metres = numpy.repeat(
            orig_height_matrix_metres, axis=0, repeats=num_examples
        )
        orig_heating_rate_matrix_w_m02 = numpy.sum(
            orig_heating_rate_matrix_k_s01 *
            orig_air_dens_matrix_kg_m03 * SPECIFIC_HEAT_J_KG01_K01 *
            orig_height_matrix_metres,
            axis=-1
        )

        new_height_matrix_metres = numpy.expand_dims(
            new_heights_metres, axis=0
        )
        new_height_matrix_metres = numpy.repeat(
            new_height_matrix_metres, axis=0, repeats=num_examples
        )
        new_heating_rate_matrix_w_m02 = numpy.sum(
            new_heating_rate_matrix_k_s01 *
            new_air_dens_matrix_kg_m03 * SPECIFIC_HEAT_J_KG01_K01 *
            new_height_matrix_metres,
            axis=-1
        )
    else:
        orig_heating_rate_matrix_w_m02 = simps(
            y=(
                orig_heating_rate_matrix_k_s01 *
                orig_air_dens_matrix_kg_m03 * SPECIFIC_HEAT_J_KG01_K01
            ),
            x=orig_heights_metres,
            axis=-1, even='avg'
        )

        new_heating_rate_matrix_w_m02 = simps(
            y=(
                new_heating_rate_matrix_k_s01 *
                new_air_dens_matrix_kg_m03 * SPECIFIC_HEAT_J_KG01_K01
            ),
            x=new_heights_metres,
            axis=-1, even='avg'
        )

    ratio_matrix = (
        orig_heating_rate_matrix_w_m02 / new_heating_rate_matrix_w_m02
    )
    # ratio_matrix[numpy.isnan(ratio_matrix)] = 0.
    ratio_matrix = numpy.expand_dims(ratio_matrix, axis=-1)
    ratio_matrix = numpy.repeat(
        ratio_matrix, axis=-1, repeats=len(new_heights_metres)
    )

    return new_heating_rate_matrix_k_day01 * ratio_matrix


def interpolate(orig_heating_rate_matrix_k_day01, orig_heights_m_agl,
                new_heights_m_agl, half_window_size_for_filter_px):
    """Interpolates heating-rate profiles to new heights.

    E = number of examples
    h = number of heights in original grid
    H = number of heights in new grid

    :param orig_heating_rate_matrix_k_day01: E-by-h numpy array of heating rates
        (Kelvins per day).
    :param orig_heights_m_agl: length-h numpy array of heights (metres above
        ground level).
    :param new_heights_m_agl: length-H numpy array of heights (metres above
        ground level).
    :param half_window_size_for_filter_px: Half-window size (pixels) for filter
        used to find local extrema.
    :return: new_heating_rate_matrix_k_day01: E-by-H numpy array of heating
        rates (Kelvins per day).
    """

    error_checking.assert_is_geq_numpy_array(
        orig_heating_rate_matrix_k_day01, 0.
    )
    error_checking.assert_is_numpy_array(
        orig_heating_rate_matrix_k_day01, num_dimensions=2
    )

    error_checking.assert_is_geq_numpy_array(orig_heights_m_agl, 0.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(orig_heights_m_agl), 0.
    )
    error_checking.assert_is_geq_numpy_array(new_heights_m_agl, 0.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(new_heights_m_agl), 0.
    )

    num_orig_heights = orig_heating_rate_matrix_k_day01.shape[1]
    error_checking.assert_is_numpy_array(
        orig_heights_m_agl,
        exact_dimensions=numpy.array([num_orig_heights], dtype=int)
    )

    error_checking.assert_is_integer(half_window_size_for_filter_px)
    error_checking.assert_is_geq(half_window_size_for_filter_px, 1)

    max_flag_matrix, max_heating_rate_matrix_k_day01 = _find_local_maxima(
        orig_heating_rate_matrix_k_day01=orig_heating_rate_matrix_k_day01,
        orig_heights_m_agl=orig_heights_m_agl,
        new_heights_m_agl=new_heights_m_agl,
        half_window_size_px=half_window_size_for_filter_px
    )

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_heating_rate_matrix_k_day01,
        axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
        fill_value='extrapolate'
    )
    new_heating_rate_matrix_k_day01 = interp_object(new_heights_m_agl)

    interp_object = interp1d(
        x=orig_heights_m_agl, y=max_heating_rate_matrix_k_day01,
        axis=-1, kind='nearest', bounds_error=False, assume_sorted=True,
        fill_value='extrapolate'
    )
    new_data_matrix_nn = interp_object(new_heights_m_agl)

    new_heating_rate_matrix_k_day01[max_flag_matrix] = (
        new_data_matrix_nn[max_flag_matrix]
    )

    top_indices = numpy.where(new_heights_m_agl > orig_heights_m_agl[-1])[0]
    new_heating_rate_matrix_k_day01[..., top_indices] = 0.

    return new_heating_rate_matrix_k_day01
