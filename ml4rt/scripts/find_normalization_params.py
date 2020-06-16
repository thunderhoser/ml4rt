"""Finds normalization parameters for radiative-transfer data."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import normalization_params

SCALAR_FIELD_NAMES = (
    example_io.SCALAR_PREDICTOR_NAMES + example_io.SCALAR_TARGET_NAMES
)

# TODO(thunderhoser): Longitude is cyclic variable.  Deal with this.

FIELD_TO_ROUNDING_BASE_SCALAR = {
    example_io.ZENITH_ANGLE_NAME: 1e-6,
    example_io.LATITUDE_NAME: 1e-4,
    example_io.LONGITUDE_NAME: 1e-4,
    example_io.ALBEDO_NAME: 1e-6,
    example_io.LIQUID_WATER_PATH_NAME: 1e-6,
    example_io.ICE_WATER_PATH_NAME: 1e-11,
    example_io.PRESSURE_NAME: 1e-1,
    example_io.TEMPERATURE_NAME: 1e-4,
    example_io.SPECIFIC_HUMIDITY_NAME: 1e-8,
    example_io.LIQUID_WATER_CONTENT_NAME: 1e-8,
    example_io.ICE_WATER_CONTENT_NAME: 1e-11,
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 1e-3,
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: 1e-3,
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 1e-3,
    example_io.SHORTWAVE_UP_FLUX_NAME: 1e-3,
    example_io.SHORTWAVE_HEATING_RATE_NAME: 1e-6
}

FIELD_TO_ROUNDING_BASE_VECTOR = {
    example_io.PRESSURE_NAME: 1e-2,
    example_io.TEMPERATURE_NAME: 1e-5,
    example_io.SPECIFIC_HUMIDITY_NAME: 1e-9,
    example_io.LIQUID_WATER_CONTENT_NAME: 1e-9,
    example_io.ICE_WATER_CONTENT_NAME: 1e-12,
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 1e-3,
    example_io.SHORTWAVE_UP_FLUX_NAME: 1e-4,
    example_io.SHORTWAVE_HEATING_RATE_NAME: 1e-7
}

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
MIN_PERCENTILE_ARG_NAME = 'min_percentile_level'
MAX_PERCENTILE_ARG_NAME = 'max_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with learning examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
FIRST_YEAR_HELP_STRING = (
    'First year in training period.  Normalization params are based only on '
    'training period.'
)
LAST_YEAR_HELP_STRING = (
    'Last year in training period.  Normalization params are based only on '
    'training period.'
)
MIN_PERCENTILE_HELP_STRING = (
    'Minimum percentile level.  The "minimum value" for each field will '
    'actually be this percentile level (in range 0...50).'
)
MAX_PERCENTILE_HELP_STRING = (
    'Maximum percentile level.  The "maximum value" for each field will '
    'actually be this percentile level (in range 50...100).'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by '
    '`normalization_params.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True,
    help=FIRST_YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True,
    help=LAST_YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_PERCENTILE_ARG_NAME, type=float, required=False, default=0.1,
    help=MIN_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.9,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_year, last_year, min_percentile_level,
         max_percentile_level, output_file_name):
    """Finds normalization parameters for radiative-transfer data.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param first_year: Same.
    :param last_year: Same.
    :param min_percentile_level: Same.
    :param max_percentile_level: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_geq(last_year, first_year)
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    num_years = len(years)
    example_file_names = [None] * num_years

    for i in range(num_years):
        example_file_names[i] = example_io.find_file(
            example_dir_name=example_dir_name, year=years[i],
            raise_error_if_missing=True
        )

    this_example_dict = example_io.read_file(example_file_names[0])
    heights_m_agl = numpy.round(
        this_example_dict[example_io.HEIGHTS_KEY]
    ).astype(int)

    orig_parameter_dict = {
        normalization_params.NUM_VALUES_KEY: 0,
        normalization_params.MEAN_VALUE_KEY: 0.,
        normalization_params.MEAN_OF_SQUARES_KEY: 0.
    }
    field_names = example_io.PREDICTOR_NAMES + example_io.TARGET_NAMES

    z_score_dict_with_height = {}
    z_score_dict_no_height = {}
    frequency_dict_with_height = {}
    frequency_dict_no_height = {}

    for this_field_name in field_names:
        z_score_dict_no_height[this_field_name] = copy.deepcopy(
            orig_parameter_dict
        )
        frequency_dict_no_height[this_field_name] = dict()

        for this_height_m_agl in heights_m_agl:
            z_score_dict_with_height[this_field_name, this_height_m_agl] = (
                copy.deepcopy(orig_parameter_dict)
            )
            frequency_dict_with_height[this_field_name, this_height_m_agl] = (
                dict()
            )

    for i in range(num_years):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
        this_example_dict = example_io.read_file(example_file_names[i])

        for this_field_name in field_names:
            print('Updating normalization params for "{0:s}"...'.format(
                this_field_name
            ))
            this_data_matrix = example_io.get_field_from_dict(
                example_dict=this_example_dict, field_name=this_field_name,
                height_m_agl=None
            )
            z_score_dict_no_height[this_field_name] = (
                normalization_params.update_z_score_params(
                    z_score_param_dict=z_score_dict_no_height[this_field_name],
                    new_data_matrix=this_data_matrix
                )
            )
            frequency_dict_no_height[this_field_name] = (
                normalization_params.update_frequency_dict(
                    frequency_dict=frequency_dict_no_height[this_field_name],
                    new_data_matrix=this_data_matrix,
                    rounding_base=FIELD_TO_ROUNDING_BASE_SCALAR[this_field_name]
                )
            )

            for this_height_m_agl in heights_m_agl:

                # TODO(thunderhoser): Could probably speed up code by not doing
                # this shit for scalar fields.
                if this_field_name in SCALAR_FIELD_NAMES:
                    z_score_dict_with_height[
                        this_field_name, this_height_m_agl
                    ] = copy.deepcopy(z_score_dict_no_height[this_field_name])

                    frequency_dict_with_height[
                        this_field_name, this_height_m_agl
                    ] = copy.deepcopy(frequency_dict_no_height[this_field_name])

                    continue

                print((
                    'Updating normalization params for "{0:s}" at {1:d} m '
                    'AGL...'
                ).format(
                    this_field_name, this_height_m_agl
                ))

                this_data_matrix = example_io.get_field_from_dict(
                    example_dict=this_example_dict, field_name=this_field_name,
                    height_m_agl=this_height_m_agl
                )

                this_dict = z_score_dict_with_height[
                    this_field_name, this_height_m_agl
                ]
                this_dict = normalization_params.update_z_score_params(
                    z_score_param_dict=this_dict,
                    new_data_matrix=this_data_matrix
                )
                z_score_dict_with_height[
                    this_field_name, this_height_m_agl
                ] = this_dict

                this_dict = frequency_dict_with_height[
                    this_field_name, this_height_m_agl
                ]
                this_dict = normalization_params.update_frequency_dict(
                    frequency_dict=this_dict,
                    new_data_matrix=this_data_matrix,
                    rounding_base=FIELD_TO_ROUNDING_BASE_VECTOR[this_field_name]
                )
                frequency_dict_with_height[
                    this_field_name, this_height_m_agl
                ] = this_dict

    norm_table_no_height = normalization_params.finalize_params(
        z_score_dict_dict=z_score_dict_no_height,
        frequency_dict_dict=frequency_dict_no_height,
        min_percentile_level=min_percentile_level,
        max_percentile_level=max_percentile_level
    )

    print((
        'Overall normalization params (not separated by height):\n{0:s}\n\n'
    ).format(
        str(norm_table_no_height)
    ))

    norm_table_with_height = normalization_params.finalize_params(
        z_score_dict_dict=z_score_dict_with_height,
        frequency_dict_dict=frequency_dict_with_height,
        min_percentile_level=min_percentile_level,
        max_percentile_level=max_percentile_level
    )

    print('Normalization params separated by height:\n{0:s}\n\n'.format(
        str(norm_table_with_height)
    ))

    normalization_params.write_file(
        pickle_file_name=output_file_name,
        norm_table_no_height=norm_table_no_height,
        norm_table_with_height=norm_table_with_height
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        min_percentile_level=getattr(INPUT_ARG_OBJECT, MIN_PERCENTILE_ARG_NAME),
        max_percentile_level=getattr(INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
