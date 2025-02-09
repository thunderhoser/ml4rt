"""Miscellaneous helper methods."""

import copy
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

EXAMPLE_IDS_KEY = 'example_id_strings'


def create_latlng_grid(
        min_latitude_deg, max_latitude_deg, latitude_spacing_deg,
        min_longitude_deg, max_longitude_deg, longitude_spacing_deg):
    """Creates lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: Minimum latitude (deg N) in grid.
    :param max_latitude_deg: Max latitude (deg N) in grid.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param min_longitude_deg: Minimum longitude (deg E) in grid.
    :param max_longitude_deg: Max longitude (deg E) in grid.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    """

    # TODO(thunderhoser): Make this handle wrap-around issues.

    min_longitude_deg = longitude_conv.convert_lng_positive_in_west(
        min_longitude_deg
    )
    max_longitude_deg = longitude_conv.convert_lng_positive_in_west(
        max_longitude_deg
    )

    min_latitude_deg = number_rounding.floor_to_nearest(
        min_latitude_deg, latitude_spacing_deg
    )
    max_latitude_deg = number_rounding.ceiling_to_nearest(
        max_latitude_deg, latitude_spacing_deg
    )
    min_longitude_deg = number_rounding.floor_to_nearest(
        min_longitude_deg, longitude_spacing_deg
    )
    max_longitude_deg = number_rounding.ceiling_to_nearest(
        max_longitude_deg, longitude_spacing_deg
    )

    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg - min_latitude_deg) / latitude_spacing_deg
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg - min_longitude_deg) / longitude_spacing_deg
    ))

    return grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )


def read_example_ids_from_netcdf(netcdf_file_name):
    """Reads example IDs from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: example_id_strings: 1-D list of example IDs.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    example_id_strings = [
        str(id) for id in
        netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
    ]
    dataset_object.close()

    return example_id_strings


def get_examples_for_inference(
        model_metadata_dict, example_file_name, num_examples, example_dir_name,
        example_id_file_name):
    """Returns examples to be used by a model at inference stage.

    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param example_file_name: [use only if you want random examples]
        Path to file with data examples (to be read by `example_io.read_file`).
    :param num_examples: [use only if you want random examples]
        Number of examples to use.  If you want to use all examples in
        `example_file_name`, leave this alone.
    :param example_dir_name: [use only if you want specific examples]
        Name of directory with data examples.  Files therein will be found by
        `example_io.find_file` and read by `example_io.read_file`.
    :param example_id_file_name: [use only if you want specific examples]
        Path to file with desired IDs.  Will be read by
        `read_example_ids_from_netcdf`.
    :return: Same output variables as `neural_net.create_data`.
    """

    error_checking.assert_is_string(example_file_name)
    use_specific_ids = example_file_name == ''

    generator_option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )

    if use_specific_ids:
        error_checking.assert_is_string(example_id_file_name)

        print('Reading desired example IDs from: "{0:s}"...'.format(
            example_id_file_name
        ))
        example_id_strings = read_example_ids_from_netcdf(example_id_file_name)

        generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
            example_dir_name
        )

        predictor_dict, target_dict = neural_net.create_data_specific_examples(
            option_dict=generator_option_dict,
            example_id_strings=example_id_strings
        )

        return predictor_dict, target_dict, example_id_strings

    error_checking.assert_is_string(example_dir_name)
    error_checking.assert_is_integer(num_examples)
    error_checking.assert_is_greater(num_examples, 0)

    example_dir_name = os.path.split(example_file_name)[0]
    year = example_io.file_name_to_year(example_file_name)
    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )

    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
        example_dir_name
    )
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    generator_option_dict[neural_net.NUM_DEEP_SUPER_LAYERS_KEY] = 0

    predictor_dict, target_dict, example_id_strings = neural_net.create_data(
        generator_option_dict
    )

    num_examples_total = len(example_id_strings)
    if num_examples >= num_examples_total:
        return predictor_dict, target_dict, example_id_strings

    good_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )
    good_indices = numpy.random.choice(
        good_indices, size=num_examples, replace=False
    )

    for this_key in predictor_dict:
        predictor_dict[this_key] = predictor_dict[this_key][good_indices, ...]
    for this_key in target_dict:
        target_dict[this_key] = target_dict[this_key][good_indices, ...]
    example_id_strings = [example_id_strings[i] for i in good_indices]

    return predictor_dict, target_dict, example_id_strings


def _handle_nonunique_example_ids(example_id_strings, first_dummy_temp_kelvins):
    """Handles non-unique example IDs.

    Specifically, this method replaces every non-unique example ID with a fake
    ID that is impossible (an extremely low or high surface temperature).

    E = number of examples

    :param example_id_strings: length-E list of example IDs.
    :param first_dummy_temp_kelvins: First dummy surface temperature.  For each
        successive fake ID, this method will increment or decrement the surface
        temperature by 10^-6 K.
    :return: example_id_strings: Same as input, except that non-unique IDs have
        been replaced with unique fake IDs.
    """

    assert first_dummy_temp_kelvins <= 101 or first_dummy_temp_kelvins >= 499

    example_id_strings_numpy = numpy.array(example_id_strings)
    unique_example_id_strings_numpy, unique_counts = numpy.unique(
        example_id_strings_numpy, return_counts=True
    )

    bad_unique_indices = numpy.where(unique_counts > 1)[0]
    dummy_temp_kelvins = first_dummy_temp_kelvins + 0.
    dummy_temp_increment = 1e-6 if first_dummy_temp_kelvins >= 499 else -1e-6

    for j in bad_unique_indices:
        this_bad_id_string = unique_example_id_strings_numpy[j]
        these_bad_indices = numpy.where(
            example_id_strings_numpy == this_bad_id_string
        )[0]

        for k in these_bad_indices:
            dummy_temp_kelvins += dummy_temp_increment
            example_id_strings_numpy[k] = (
                '{0:s}_temp-10m-kelvins={1:010.6f}'
            ).format(
                '_'.join(example_id_strings_numpy[k].split('_')[:-1]),
                dummy_temp_kelvins
            )

    example_id_strings = example_id_strings_numpy.tolist()
    assert len(example_id_strings) == len(set(example_id_strings))

    return example_id_strings


def get_raw_examples(
        example_file_name, num_examples, example_dir_name,
        example_id_file_name, ignore_sfc_temp_in_example_id=False,
        allow_missing_examples=False):
    """Returns raw examples.

    The difference between `get_raw_examples` and `get_examples_for_inference`
    is that `get_raw_examples` returns examples in their raw form, *not*
    pre-processed to be fed through a model for inference.

    :param example_file_name: See doc for `get_examples_for_inference`.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param ignore_sfc_temp_in_example_id: Boolean flag.  If True, will ignore
        surface temperature when matching example IDs.
    :param allow_missing_examples: Boolean flag.  If True, will allow missing
        examples.
    :return: example_dict: See doc for `example_io.read_file`.
    :return: found_example_flags: length-E numpy array of Boolean flags
        indicating which desired examples were found, where E = number of
        example IDs read from `example_id_file_name`.
    """

    error_checking.assert_is_string(example_file_name)
    use_specific_ids = example_file_name == ''

    if use_specific_ids:
        error_checking.assert_is_string(example_id_file_name)
        error_checking.assert_is_boolean(ignore_sfc_temp_in_example_id)
        error_checking.assert_is_boolean(allow_missing_examples)

        print('Reading desired example IDs from: "{0:s}"...'.format(
            example_id_file_name
        ))
        example_id_strings = read_example_ids_from_netcdf(example_id_file_name)

        metadata_dict = example_utils.parse_example_ids(example_id_strings)
        valid_times_unix_sec = metadata_dict[example_utils.VALID_TIMES_KEY]

        if ignore_sfc_temp_in_example_id:
            example_id_strings = [
                '_'.join(s.split('_')[:-1]) + '_temp-10m-kelvins=200.000000'
                for s in example_id_strings
            ]
            example_id_strings = _handle_nonunique_example_ids(
                example_id_strings=example_id_strings,
                first_dummy_temp_kelvins=100.
            )

        example_file_names = example_io.find_many_files(
            directory_name=example_dir_name,
            first_time_unix_sec=numpy.min(valid_times_unix_sec),
            last_time_unix_sec=numpy.max(valid_times_unix_sec)
        )

        num_files = len(example_file_names)
        example_dicts = [dict()] * num_files

        for i in range(num_files):
            print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
            example_dicts[i] = example_io.read_file(example_file_names[i])

        example_dict = example_utils.concat_examples(example_dicts)

        if ignore_sfc_temp_in_example_id:
            example_dict[example_utils.EXAMPLE_IDS_KEY] = [
                '_'.join(s.split('_')[:-1]) + '_temp-10m-kelvins=200.000000'
                for s in example_dict[example_utils.EXAMPLE_IDS_KEY]
            ]
            example_dict[example_utils.EXAMPLE_IDS_KEY] = (
                _handle_nonunique_example_ids(
                    example_id_strings=
                    example_dict[example_utils.EXAMPLE_IDS_KEY],
                    first_dummy_temp_kelvins=500.
                )
            )

        good_indices = example_utils.find_examples(
            all_id_strings=example_dict[example_utils.EXAMPLE_IDS_KEY],
            desired_id_strings=example_id_strings,
            allow_missing=allow_missing_examples
        )

        found_example_flags = good_indices >= 0
        good_indices[good_indices < 0] = len(good_indices) - 1

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_indices
        )
    else:
        error_checking.assert_is_string(example_dir_name)
        error_checking.assert_is_integer(num_examples)
        error_checking.assert_is_greater(num_examples, 0)

        print('Reading data from: "{0:s}"...'.format(example_file_name))
        example_dict = example_io.read_file(example_file_name)

        num_examples_total = len(example_dict[example_utils.VALID_TIMES_KEY])
        desired_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )

        if num_examples < num_examples_total:
            desired_indices = numpy.random.choice(
                desired_indices, size=num_examples, replace=False
            )

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=desired_indices
        )

        num_examples = len(example_dict[example_utils.VALID_TIMES_KEY])
        found_example_flags = numpy.full(num_examples, 1, dtype=bool)

    return example_dict, found_example_flags


def find_best_and_worst_predictions(bias_matrix_3d, absolute_error_matrix_3d,
                                    num_examples_per_set):
    """Finds best and worst predictions.

    E = total number of examples
    H = number of heights
    W = number of wavelengths
    e = number of examples per set

    :param bias_matrix_3d: E-by-H-by-W numpy array of biases (predicted minus
        actual).
    :param absolute_error_matrix_3d: E-by-H-by-W numpy array of absolute errors.
    :param num_examples_per_set: Number of examples per set.
    :return: high_bias_indices: length-e numpy array with indices of high-bias
        examples.
    :return: low_bias_indices: length-e numpy array with indices of low-bias
        examples.
    :return: low_abs_error_indices: length-e numpy array with indices of
        low-absolute-error examples.
    """

    max_bias_by_example = numpy.max(bias_matrix_3d, axis=(1, 2))
    sort_indices = numpy.argsort(-1 * max_bias_by_example)
    high_bias_indices = sort_indices[:num_examples_per_set]

    for i in range(num_examples_per_set):
        print('{0:d}th-greatest positive bias = {1:f}'.format(
            i + 1, max_bias_by_example[high_bias_indices[i]]
        ))

    print(SEPARATOR_STRING)

    min_bias_by_example = numpy.min(bias_matrix_3d, axis=(1, 2))
    sort_indices = numpy.argsort(min_bias_by_example)
    low_bias_indices = sort_indices[:num_examples_per_set]

    for i in range(num_examples_per_set):
        print('{0:d}th-greatest negative bias = {1:f}'.format(
            i + 1, min_bias_by_example[low_bias_indices[i]]
        ))

    print(SEPARATOR_STRING)

    max_abs_error_by_example = numpy.max(absolute_error_matrix_3d, axis=(1, 2))
    sort_indices = numpy.argsort(max_abs_error_by_example)
    low_abs_error_indices = sort_indices[:num_examples_per_set]

    for i in range(num_examples_per_set):
        print('{0:d}th-smallest absolute error = {1:f}'.format(
            i + 1, max_abs_error_by_example[low_abs_error_indices[i]]
        ))

    return high_bias_indices, low_bias_indices, low_abs_error_indices
