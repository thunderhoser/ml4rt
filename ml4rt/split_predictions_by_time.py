"""Splits predictions by 'time'.

Specifically, this script splits predictions in three different ways:

- By time of year (quantified by month)
- By time of day (quantified by solar zenith angle)
- By albedo (proxy for time of year)
"""

import os
import sys
import copy
import argparse
import numpy
from scipy.integrate import simps

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import example_io
import example_utils

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = numpy.pi / 2
MAX_SHORTWAVE_SFC_DOWN_FLUX_W_M02 = 1200.
MAX_AEROSOL_OPTICAL_DEPTH = 1.8

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
NUM_ALBEDO_BINS_ARG_NAME = 'num_albedo_bins'
NUM_FLUX_BINS_ARG_NAME = 'num_shortwave_sfc_down_flux_bins'
NUM_AOD_BINS_ARG_NAME = 'num_aod_bins'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all times of day/year.  '
    'Will be read by `prediction_io.read_file`.'
)
NUM_ANGLE_BINS_HELP_STRING = 'Number of bins for zenith angle.'
NUM_ALBEDO_BINS_HELP_STRING = 'Number of bins for albedo.'
NUM_FLUX_BINS_HELP_STRING = (
    'Number of bins for shortwave surface downwelling flux.  If you do not want'
    ' to split by shortwave surface downwelling flux, make this argument <= 0.'
)
NUM_AOD_BINS_HELP_STRING = (
    'Number of bins for aerosol optical depth (AOD).  If you do not want to '
    'split by AOD, make this argument <= 0.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files, used only if `{0:s}` > 0.  Aerosol '
    'optical depths will be computed from aerosol-extinction profiles in these '
    'files.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Temporally split predictions will be written '
    'here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ALBEDO_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_ALBEDO_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FLUX_BINS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_AOD_BINS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_AOD_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, num_zenith_angle_bins, num_albedo_bins,
         num_shortwave_sfc_down_flux_bins, num_aod_bins, example_dir_name,
         output_dir_name):
    """Splits predictions by time of day and time of year.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param num_zenith_angle_bins: Same.
    :param num_albedo_bins: Same.
    :param num_shortwave_sfc_down_flux_bins: Same.
    :param num_aod_bins: Same.
    :param example_dir_name: Same.
    :param output_dir_name: Same.
    """

    # Process input args.
    if num_shortwave_sfc_down_flux_bins <= 0:
        num_shortwave_sfc_down_flux_bins = None
    if num_aod_bins <= 0:
        num_aod_bins = None

    error_checking.assert_is_geq(num_zenith_angle_bins, 3)
    error_checking.assert_is_geq(num_albedo_bins, 3)
    if num_shortwave_sfc_down_flux_bins is not None:
        error_checking.assert_is_geq(num_shortwave_sfc_down_flux_bins, 3)
    if num_aod_bins is not None:
        error_checking.assert_is_geq(num_aod_bins, 3)

    edge_zenith_angles_rad = numpy.linspace(
        MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
        num=num_zenith_angle_bins + 1, dtype=float
    )
    min_zenith_angles_rad = edge_zenith_angles_rad[:-1]
    max_zenith_angles_rad = edge_zenith_angles_rad[1:]

    edge_albedos = numpy.linspace(0, 1, num=num_albedo_bins + 1, dtype=float)
    min_albedos = edge_albedos[:-1]
    max_albedos = edge_albedos[1:]

    # Read data.
    print('Reading data from: "{0:s}"...\n'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)

    # Split by solar zenith angle.
    for k in range(num_zenith_angle_bins):
        this_prediction_dict = prediction_io.subset_by_zenith_angle(
            prediction_dict=copy.deepcopy(prediction_dict),
            min_zenith_angle_rad=min_zenith_angles_rad[k],
            max_zenith_angle_rad=max_zenith_angles_rad[k]
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, zenith_angle_bin=k,
            raise_error_if_missing=False
        )
        print((
            'Writing {0:d} examples (with zenith angles {1:.4f}...{2:.4f} rad) '
            'to: "{3:s}"...'
        ).format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            min_zenith_angles_rad[k], max_zenith_angles_rad[k],
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )

    print('\n')

    # Split by albedo.
    for k in range(num_albedo_bins):
        this_prediction_dict = prediction_io.subset_by_albedo(
            prediction_dict=copy.deepcopy(prediction_dict),
            min_albedo=min_albedos[k], max_albedo=max_albedos[k]
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, albedo_bin=k,
            raise_error_if_missing=False
        )
        print((
            'Writing {0:d} examples (with albedos {1:.4f}...{2:.4f}) '
            'to: "{3:s}"...'
        ).format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            min_albedos[k], max_albedos[k], this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )

    print('\n')

    # Split by month.
    for k in range(1, 13):
        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=copy.deepcopy(prediction_dict), desired_month=k
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, month=k,
            raise_error_if_missing=False
        )
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )

    print('\n')

    # Split by shortwave surface downwelling flux.
    if num_shortwave_sfc_down_flux_bins is not None:
        edge_fluxes_w_m02 = numpy.linspace(
            0, MAX_SHORTWAVE_SFC_DOWN_FLUX_W_M02,
            num=num_shortwave_sfc_down_flux_bins + 1, dtype=float
        )
        min_fluxes_w_m02 = edge_fluxes_w_m02[:-1]
        max_fluxes_w_m02 = edge_fluxes_w_m02[1:]
        max_fluxes_w_m02[-1] = numpy.inf

        for k in range(num_shortwave_sfc_down_flux_bins):
            this_prediction_dict = (
                prediction_io.subset_by_shortwave_sfc_down_flux(
                    prediction_dict=copy.deepcopy(prediction_dict),
                    min_flux_w_m02=min_fluxes_w_m02[k],
                    max_flux_w_m02=max_fluxes_w_m02[k]
                )
            )

            this_output_file_name = prediction_io.find_file(
                directory_name=output_dir_name, shortwave_sfc_down_flux_bin=k,
                raise_error_if_missing=False
            )
            print((
                'Writing {0:d} examples (with shortwave surface downwelling '
                'fluxes of {1:.4f}...{2:.4f} W m^-2) to: "{3:s}"...'
            ).format(
                len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
                min_fluxes_w_m02[k], max_fluxes_w_m02[k], this_output_file_name
            ))

            prediction_io.write_file(
                netcdf_file_name=this_output_file_name,
                scalar_target_matrix=
                this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
                vector_target_matrix=
                this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
                scalar_prediction_matrix=
                this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
                vector_prediction_matrix=
                this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
                heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
                example_id_strings=
                this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
                model_file_name=
                this_prediction_dict[prediction_io.MODEL_FILE_KEY],
                normalization_file_name=
                this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
            )

        print('\n')

    if num_aod_bins is None:
        return

    valid_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_id_strings = []
    aerosol_extinction_matrix_metres01 = numpy.array([])
    height_matrix_m_agl = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        this_extinction_matrix_metres01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.AEROSOL_EXTINCTION_NAME
        )

        if aerosol_extinction_matrix_metres01.size == 0:
            aerosol_extinction_matrix_metres01 = (
                this_extinction_matrix_metres01 + 0.
            )
        else:
            aerosol_extinction_matrix_metres01 = numpy.concatenate((
                aerosol_extinction_matrix_metres01,
                this_extinction_matrix_metres01
            ), axis=0)

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            this_height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.HEIGHT_NAME
            )

            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = this_height_matrix_m_agl + 0.
            else:
                height_matrix_m_agl = numpy.concatenate(
                    (height_matrix_m_agl, this_height_matrix_m_agl), axis=0
                )
        else:
            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = (
                    this_example_dict[example_utils.HEIGHTS_KEY] + 0.
                )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )
    del example_id_strings

    aerosol_extinction_matrix_metres01 = (
        aerosol_extinction_matrix_metres01[desired_indices, :]
    )

    if len(height_matrix_m_agl.shape) == 2:
        height_matrix_m_agl = height_matrix_m_agl[desired_indices, :]
        num_examples = aerosol_extinction_matrix_metres01.shape[0]
        aerosol_optical_depths = numpy.full(num_examples, numpy.nan)
        print('\n')

        for i in range(num_examples):
            if numpy.mod(i, 1000) == 0:
                print((
                    'Have computed aerosol optical depth for {0:d} of {1:d} '
                    'profiles...'
                ).format(
                    i, num_examples
                ))

            aerosol_optical_depths[i] = simps(
                y=aerosol_extinction_matrix_metres01[i, :],
                x=height_matrix_m_agl[i, :],
                even='avg'
            )

        print((
            'Have computed aerosol optical depth for all {0:d} profiles!\n'
        ).format(
            num_examples
        ))
    else:
        aerosol_optical_depths = simps(
            y=aerosol_extinction_matrix_metres01, x=height_matrix_m_agl,
            axis=-1, even='avg'
        )

    edge_aerosol_optical_depths = numpy.linspace(
        0, MAX_AEROSOL_OPTICAL_DEPTH,
        num=num_aod_bins + 1, dtype=float
    )
    min_aerosol_optical_depths = edge_aerosol_optical_depths[:-1]
    max_aerosol_optical_depths = edge_aerosol_optical_depths[1:]
    max_aerosol_optical_depths[-1] = numpy.inf

    for k in range(num_aod_bins):
        these_indices = numpy.where(numpy.logical_and(
            aerosol_optical_depths >= min_aerosol_optical_depths[k],
            aerosol_optical_depths <= max_aerosol_optical_depths[k]
        ))[0]

        this_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=these_indices
        )

        this_output_file_name = prediction_io.find_file(
            directory_name=output_dir_name, aerosol_optical_depth_bin=k,
            raise_error_if_missing=False
        )
        print((
            'Writing {0:d} examples (with aerosol optical depths of '
            '{1:.4f}...{2:.4f}) to: "{3:s}"...'
        ).format(
            len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
            min_aerosol_optical_depths[k],
            max_aerosol_optical_depths[k],
            this_output_file_name
        ))

        prediction_io.write_file(
            netcdf_file_name=this_output_file_name,
            scalar_target_matrix=
            this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
            vector_target_matrix=
            this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
            scalar_prediction_matrix=
            this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
            vector_prediction_matrix=
            this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
            heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
            example_id_strings=
            this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
            model_file_name=
            this_prediction_dict[prediction_io.MODEL_FILE_KEY],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        num_albedo_bins=getattr(INPUT_ARG_OBJECT, NUM_ALBEDO_BINS_ARG_NAME),
        num_shortwave_sfc_down_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_FLUX_BINS_ARG_NAME
        ),
        num_aod_bins=getattr(INPUT_ARG_OBJECT, NUM_AOD_BINS_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
