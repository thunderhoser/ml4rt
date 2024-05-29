"""Normalizes learning examples."""

import os.path
import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import normalization
from ml4rt.utils import example_utils

METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_example_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
NORMALIZE_PREDICTORS_ARG_NAME = 'normalize_predictors'
NORMALIZE_SCALAR_TARGETS_ARG_NAME = 'normalize_scalar_targets'
NORMALIZE_VECTOR_TARGETS_ARG_NAME = 'normalize_vector_targets'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to file with unnormalized examples.  Will be read by '
    '`example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to file with normalization params (will be read by '
    '`normalization.read_params`).'
)
NORMALIZE_PREDICTORS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize predictor variables.'
)
NORMALIZE_SCALAR_TARGETS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize scalar target variables.'
)
NORMALIZE_VECTOR_TARGETS_HELP_STRING = (
    'Boolean flag.  If 1, will normalize vector target variables.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized examples will be written here by '
    '`example_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZE_PREDICTORS_ARG_NAME, type=int, required=True,
    help=NORMALIZE_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZE_SCALAR_TARGETS_ARG_NAME, type=int, required=True,
    help=NORMALIZE_SCALAR_TARGETS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZE_VECTOR_TARGETS_ARG_NAME, type=int, required=True,
    help=NORMALIZE_VECTOR_TARGETS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_file_name, normalization_file_name,
         normalize_predictors, normalize_scalar_targets,
         normalize_vector_targets, output_example_dir_name):
    """Normalizes learning examples.

    This is effectively the main method.

    :param input_example_file_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param normalize_predictors: Same.
    :param normalize_scalar_targets: Same.
    :param normalize_vector_targets: Same.
    :param output_example_dir_name: Same.
    """

    print('Reading unnormalized examples from: "{0:s}"...'.format(
        input_example_file_name
    ))
    example_dict = example_io.read_file(
        netcdf_file_name=input_example_file_name,
        max_shortwave_heating_k_day01=numpy.inf,
        min_longwave_heating_k_day01=-1 * numpy.inf,
        max_longwave_heating_k_day01=numpy.inf
    )

    normalization_metadata_dict = (
        example_dict[example_utils.NORMALIZATION_METADATA_KEY]
    )
    assert (
        normalization_metadata_dict[example_io.NORMALIZATION_FILE_KEY] is None
    )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = normalization.read_params(normalization_file_name)

    normalization.normalize_data(
        example_dict=example_dict,
        normalization_param_table_xarray=norm_param_table_xarray,
        apply_to_predictors=normalize_predictors,
        apply_to_scalar_targets=normalize_scalar_targets,
        apply_to_vector_targets=normalize_vector_targets
    )

    for j in range(len(example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY])):
        these_values = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j]
        )
        print('Mean normalized {0:s} = {1:.4g}'.format(
            example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
            numpy.mean(these_values)
        ))

    for j in range(len(example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY])):
        for h in range(len(example_dict[example_utils.HEIGHTS_KEY])):
            these_values = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=
                example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY][j],
                height_m_agl=example_dict[example_utils.HEIGHTS_KEY][h]
            )
            print('Mean normalized {0:s} at {1:.0f} m AGL = {2:.4g}'.format(
                example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
                example_dict[example_utils.HEIGHTS_KEY][h],
                numpy.mean(these_values)
            ))

    for j in range(len(example_dict[example_utils.SCALAR_TARGET_NAMES_KEY])):
        for w in range(len(example_dict[example_utils.TARGET_WAVELENGTHS_KEY])):
            these_values = example_utils.get_field_from_dict(
                example_dict=example_dict,
                field_name=
                example_dict[example_utils.SCALAR_TARGET_NAMES_KEY][j],
                target_wavelength_metres=
                example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w]
            )
            print('Mean normalized {0:s} at {1:.2f} microns = {2:.4g}'.format(
                example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
                METRES_TO_MICRONS *
                example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                numpy.mean(these_values)
            ))

    for j in range(len(example_dict[example_utils.VECTOR_TARGET_NAMES_KEY])):
        for w in range(len(example_dict[example_utils.TARGET_WAVELENGTHS_KEY])):
            for h in range(len(example_dict[example_utils.HEIGHTS_KEY])):
                these_values = example_utils.get_field_from_dict(
                    example_dict=example_dict,
                    field_name=
                    example_dict[example_utils.VECTOR_TARGET_NAMES_KEY][j],
                    target_wavelength_metres=
                    example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                    height_m_agl=example_dict[example_utils.HEIGHTS_KEY][h]
                )
                print((
                    'Mean normalized {0:s} at {1:.0f} m AGL and {2:.2f} '
                    'microns = {3:.4g}'
                ).format(
                    example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY][j],
                    example_dict[example_utils.HEIGHTS_KEY][h],
                    METRES_TO_MICRONS *
                    example_dict[example_utils.TARGET_WAVELENGTHS_KEY][w],
                    numpy.mean(these_values)
                ))

    normalization_metadata_dict = {
        example_io.NORMALIZATION_FILE_KEY: normalization_file_name,
        example_io.NORMALIZE_PREDICTORS_KEY: normalize_predictors,
        example_io.NORMALIZE_SCALAR_TARGETS_KEY: normalize_scalar_targets,
        example_io.NORMALIZE_VECTOR_TARGETS_KEY: normalize_vector_targets
    }
    example_dict[example_utils.NORMALIZATION_METADATA_KEY] = (
        normalization_metadata_dict
    )

    output_example_file_name = '{0:s}/{1:s}'.format(
        output_example_dir_name, os.path.split(input_example_file_name)[1]
    )
    print('Writing normalized examples to: "{0:s}"...'.format(
        output_example_file_name
    ))
    example_io.write_file(
        example_dict=example_dict, netcdf_file_name=output_example_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        normalize_predictors=bool(getattr(
            INPUT_ARG_OBJECT, NORMALIZE_PREDICTORS_ARG_NAME
        )),
        normalize_scalar_targets=bool(getattr(
            INPUT_ARG_OBJECT, NORMALIZE_SCALAR_TARGETS_ARG_NAME
        )),
        normalize_vector_targets=bool(getattr(
            INPUT_ARG_OBJECT, NORMALIZE_VECTOR_TARGETS_ARG_NAME
        )),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
