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


def subset_examples(indices_to_keep, num_examples_to_keep, num_examples_total):
    """Subsets examples.

    :param indices_to_keep: 1-D numpy array with indices to keep.  If None, will
        use `num_examples_to_keep` instead.
    :param num_examples_to_keep: Number of examples to keep.  If None, will use
        `indices_to_keep` instead.
    :param num_examples_total: Total number of examples available.
    :return: indices_to_keep: See input doc.
    """

    if len(indices_to_keep) == 1 and indices_to_keep[0] < 0:
        indices_to_keep = None
    if indices_to_keep is not None:
        num_examples_to_keep = None
    if num_examples_to_keep < 1:
        num_examples_to_keep = None

    if indices_to_keep is None and num_examples_to_keep is None:
        num_examples_to_keep = num_examples_total

    if indices_to_keep is not None:
        error_checking.assert_is_geq_numpy_array(indices_to_keep, 0)
        error_checking.assert_is_less_than_numpy_array(
            indices_to_keep, num_examples_total
        )

        return indices_to_keep

    indices_to_keep = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )

    if num_examples_to_keep >= num_examples_total:
        return indices_to_keep

    return numpy.random.choice(
        indices_to_keep, size=num_examples_to_keep, replace=False
    )


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

    # TODO(thunderhosder): Make this handle wrap-around issues.

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
    :return: Same output variables as `neural_net.data_generator`.
    """

    error_checking.assert_is_string(example_file_name)
    use_specific_ids = example_file_name == ''

    if use_specific_ids:
        error_checking.assert_is_string(example_id_file_name)

        print('Reading desired example IDs from: "{0:s}"...'.format(
            example_id_file_name
        ))
        example_id_strings = read_example_ids_from_netcdf(example_id_file_name)
        num_examples_per_batch = len(example_id_strings)
    else:
        error_checking.assert_is_string(example_dir_name)
        error_checking.assert_is_integer(num_examples)
        error_checking.assert_is_greater(num_examples, 0)

        example_dir_name = os.path.split(example_file_name)[0]
        year = example_io.file_name_to_year(example_file_name)
        first_time_unix_sec, last_time_unix_sec = (
            time_conversion.first_and_last_times_in_year(year)
        )

        this_example_dict = example_io.read_file(example_file_name)
        num_examples_per_batch = len(
            this_example_dict[example_utils.VALID_TIMES_KEY]
        )

    generator_option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = num_examples_per_batch

    if use_specific_ids:
        generator = neural_net.data_generator_specific_examples(
            option_dict=generator_option_dict,
            net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
            example_id_strings=example_id_strings
        )
    else:
        generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
        generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

        generator = neural_net.data_generator(
            option_dict=generator_option_dict, for_inference=True,
            net_type_string=model_metadata_dict[neural_net.NET_TYPE_KEY],
            is_loss_constrained_mse=False
        )

    print(SEPARATOR_STRING)

    if use_specific_ids:
        predictor_matrix, target_array = next(generator)
    else:
        predictor_matrix, target_array, example_id_strings = next(generator)

        good_indices = subset_examples(
            indices_to_keep=numpy.array([-1], dtype=int),
            num_examples_to_keep=num_examples,
            num_examples_total=len(example_id_strings)
        )

        predictor_matrix = predictor_matrix[good_indices, ...]
        example_id_strings = [example_id_strings[i] for i in good_indices]

        if isinstance(target_array, list):
            target_array = [t[good_indices, ...] for t in target_array]
        else:
            target_array = target_array[good_indices, ...]

    return predictor_matrix, target_array, example_id_strings


def get_raw_examples(
        example_file_name, num_examples, example_dir_name,
        example_id_file_name):
    """Returns raw examples.

    The difference between `get_raw_examples` and `get_examples_for_inference`
    is that `get_raw_examples` returns examples in their raw form, *not*
    pre-processed to be fed through a model for inference.

    :param example_file_name: See doc for `get_examples_for_inference`.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :return: example_dict: See doc for `example_io.read_file`.
    """

    error_checking.assert_is_string(example_file_name)
    use_specific_ids = example_file_name == ''

    if use_specific_ids:
        error_checking.assert_is_string(example_id_file_name)

        print('Reading desired example IDs from: "{0:s}"...'.format(
            example_id_file_name
        ))
        example_id_strings = read_example_ids_from_netcdf(example_id_file_name)

        valid_times_unix_sec = example_utils.parse_example_ids(
            example_id_strings
        )[example_utils.VALID_TIMES_KEY]

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

        good_indices = example_utils.find_examples(
            all_id_strings=example_dict[example_utils.EXAMPLE_IDS_KEY],
            desired_id_strings=example_id_strings, allow_missing=False
        )

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_indices
        )
    else:
        error_checking.assert_is_string(example_dir_name)
        error_checking.assert_is_integer(num_examples)
        error_checking.assert_is_greater(num_examples, 0)

        print('Reading data from: "{0:s}"...'.format(example_file_name))
        example_dict = example_io.read_file(example_file_name)
        example_dict = example_utils.reduce_sample_size(
            example_dict=example_dict, num_examples_to_keep=num_examples
        )[0]

    return example_dict
