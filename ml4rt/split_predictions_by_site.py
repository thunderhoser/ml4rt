"""Splits predictions by site (point location)."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import number_rounding
import longitude_conversion as lng_conversion
import prediction_io
import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

LATLNG_TOLERANCE_DEG = 1e-4

SITE_NAME_TO_LATLNG = {
    'juelich_germany': numpy.array([50.8786, 6.4798]),
    'santa_barbara_ca': numpy.array([34.4773, 240.0771]),
    'bodega_bay_ca': numpy.array([38.2638, 236.9027]),
    'forks_wa': numpy.array([47.9740, 235.6823]),
    'owego_ny': numpy.array([42.0287, 283.7354]),
    'tupper_lake_ny': numpy.array([44.2570, 285.4905]),
    'belle_mina_al': numpy.array([34.6806, 273.0676]),
    'ny_alesund_norway_land': numpy.array([78.9158, 11.8892]),
    'eureka_nu_ocean': numpy.array([79.7353, 273.3608]),
    'barrow_ak_ocean': numpy.array([71.4970, 202.9377]),
    'lamont_ok': numpy.array([36.5693, 262.5687]),
    'azores_arm_site': numpy.array([39.0827, -28.0923]),
    'villum_greenland_land': numpy.array([81.4876, -16.1931]),
    # 'summit_greenland_land': numpy.array([72.5790, -38.3127]),
    'alert_nu_land': numpy.array([82.4291, -61.9947]),
    'yopp_arctic_ocean_site3': numpy.array([81.0731, -134.8826]),
    'sheba': numpy.array([76.0152, -164.7252]),
    'cherskii_russia_ocean': numpy.array([69.7849, 161.2023]),
    'tiksi_russia_ocean': numpy.array([71.6260, 129.7669]),
    'pallas_sodankyla_russia_land': numpy.array([67.8995, 24.1929]),
    'lindenberg_germany': numpy.array([52.2341, 14.1223]),
    'fino3_tower_north_sea': numpy.array([55.1932, 7.1276]),
    'yopp_arctic_ocean_site2': numpy.array([89.9983, 164.0000]),
    'yopp_arctic_ocean_site1': numpy.array([84.9823, 10.0999]),
    'bishop_grenada': numpy.array([12.049037, 298.225193]),
    'rohlsen_us_virgin_islands': numpy.array([17.658863, 295.181625]),
    'san_juan_pr': numpy.array([18.428686, 294.029716]),
    'hilo_hi': numpy.array([19.711315, 204.901398]),
    'guantanamo_bay': numpy.array([19.843439, 284.852074]),
    'honolulu_hi': numpy.array([21.326672, 202.113266]),
    'perdido_oil_rig': numpy.array([26.133038, 265.124474])
}

for this_key in SITE_NAME_TO_LATLNG:
    SITE_NAME_TO_LATLNG[this_key][1] = (
        lng_conversion.convert_lng_positive_in_west(
            SITE_NAME_TO_LATLNG[this_key][1]
        )
    )

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all sites.  Will be read by'
    ' `prediction_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of top-level output directory.  Split predictions (one file per site)'
    ' will be written here by `prediction_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, top_output_dir_name):
    """Splits predictions by site (point location).

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param top_output_dir_name: Same.
    :raises: ValueError: if any example cannot be assigned to a site.
    """

    # Read data.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)
    example_metadata_dict = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )

    example_latitudes_deg_n = number_rounding.round_to_nearest(
        example_metadata_dict[example_utils.LATITUDES_KEY],
        LATLNG_TOLERANCE_DEG
    )
    example_longitudes_deg_e = number_rounding.round_to_nearest(
        example_metadata_dict[example_utils.LONGITUDES_KEY],
        LATLNG_TOLERANCE_DEG
    )
    example_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        example_longitudes_deg_e
    )

    num_examples = len(example_latitudes_deg_n)
    example_written_flags = numpy.full(num_examples, False, dtype=bool)

    site_names = list(SITE_NAME_TO_LATLNG.keys())
    num_sites = len(site_names)

    for j in range(num_sites):
        this_site_latitude_deg_n = SITE_NAME_TO_LATLNG[site_names[j]][0]
        this_site_longitude_deg_e = SITE_NAME_TO_LATLNG[site_names[j]][1]

        these_indices = numpy.where(numpy.logical_and(
            numpy.absolute(example_latitudes_deg_n - this_site_latitude_deg_n)
            <= LATLNG_TOLERANCE_DEG,
            numpy.absolute(example_longitudes_deg_e - this_site_longitude_deg_e)
            <= LATLNG_TOLERANCE_DEG
        ))[0]

        this_prediction_dict = prediction_io.subset_by_index(
            prediction_dict=copy.deepcopy(prediction_dict),
            desired_indices=these_indices
        )

        this_output_file_name = '{0:s}/{1:s}/predictions.nc'.format(
            top_output_dir_name, site_names[j]
        )
        print('Writing {0:d} examples to: "{1:s}"...'.format(
            len(these_indices), this_output_file_name
        ))

        if len(these_indices) == 0:
            continue

        example_written_flags[these_indices] = True

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
            isotonic_model_file_name=
            this_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
            uncertainty_calib_model_file_name=this_prediction_dict[
                prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
            ],
            normalization_file_name=
            this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
        )

    if numpy.all(example_written_flags):
        return

    # bad_latitudes_deg_n = (
    #     example_latitudes_deg_n[example_written_flags == False]
    # )
    # bad_longitudes_deg_e = (
    #     example_longitudes_deg_e[example_written_flags == False]
    # )
    # bad_coord_matrix = numpy.transpose(numpy.vstack((
    #     bad_latitudes_deg_n, bad_longitudes_deg_e
    # )))
    # bad_coord_matrix = numpy.unique(bad_coord_matrix, axis=0)
    # print(bad_coord_matrix)

    error_string = (
        '{0:d} of {1:d} examples could not be assigned to a site.  This is a '
        'BIG PROBLEM.'
    ).format(
        numpy.sum(example_written_flags == False), num_examples
    )

    raise ValueError(error_string)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
