"""Scratch space."""

import numpy
from ml4rt.io import example_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

RADIATION_FILE_NAME = (
    '/home/ryan.lagerquist/dave_turner_rt_stuff/output_file.2018.cdf'
)

example_dict = example_io.read_file(RADIATION_FILE_NAME)
# print(example_dict)

scalar_predictor_names = example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY]
vector_predictor_names = example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]

for k in range(len(scalar_predictor_names)):
    this_min_value = numpy.min(
        example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY][:, k]
    )
    this_mean_value = numpy.mean(
        example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY][:, k]
    )
    this_max_value = numpy.max(
        example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY][:, k]
    )

    print('Min/mean/max for "{0:s}" = {1:f}, {2:f}, {3:f}'.format(
        scalar_predictor_names[k], this_min_value, this_mean_value,
        this_max_value
    ))

for k in range(len(scalar_target_names)):
    this_min_value = numpy.min(
        example_dict[example_io.SCALAR_TARGET_VALS_KEY][:, k]
    )
    this_mean_value = numpy.mean(
        example_dict[example_io.SCALAR_TARGET_VALS_KEY][:, k]
    )
    this_max_value = numpy.max(
        example_dict[example_io.SCALAR_TARGET_VALS_KEY][:, k]
    )

    print('Min/mean/max for "{0:s}" = {1:f}, {2:f}, {3:f}'.format(
        scalar_target_names[k], this_min_value, this_mean_value,
        this_max_value
    ))

print(SEPARATOR_STRING)
heights_m_agl = example_dict[example_io.HEIGHTS_KEY]

for k in range(len(vector_predictor_names)):
    these_min_values = numpy.min(
        example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY][..., k], axis=0
    )
    these_mean_values = numpy.mean(
        example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY][..., k], axis=0
    )
    these_max_values = numpy.max(
        example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY][..., k], axis=0
    )

    for i in range(len(heights_m_agl)):
        print((
            'Min/mean/max for "{0:s}" at {1:f} m AGL = {2:f}, {3:f}, {4:f}'
        ).format(
            vector_predictor_names[k], heights_m_agl[i], these_min_values[i],
            these_mean_values[i], these_max_values[i]
        ))

    print(SEPARATOR_STRING)

for k in range(len(vector_target_names)):
    these_min_values = numpy.min(
        example_dict[example_io.VECTOR_TARGET_VALS_KEY][..., k], axis=0
    )
    these_mean_values = numpy.mean(
        example_dict[example_io.VECTOR_TARGET_VALS_KEY][..., k], axis=0
    )
    these_max_values = numpy.max(
        example_dict[example_io.VECTOR_TARGET_VALS_KEY][..., k], axis=0
    )

    for i in range(len(heights_m_agl)):
        print((
            'Min/mean/max for "{0:s}" at {1:f} m AGL = {2:f}, {3:f}, {4:f}'
        ).format(
            vector_target_names[k], heights_m_agl[i], these_min_values[i],
            these_mean_values[i], these_max_values[i]
        ))

    print(SEPARATOR_STRING)
