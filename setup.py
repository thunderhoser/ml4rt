"""Setup file for ml4rt."""

from setuptools import setup

PACKAGE_NAMES = [
    'ml4rt', 'ml4rt.io', 'ml4rt.utils'
]
KEYWORDS = [
    'machine learning', 'deep learning', 'artificial intelligence',
    'data mining', 'weather', 'meteorology', 'thunderstorm', 'wind', 'tornado'
]
SHORT_DESCRIPTION = 'Machine learning for radiative transfer.'
LONG_DESCRIPTION = (
    'ml4rt is an end-to-end machine-learning library for emulating physical '
    'radiative-transfer models.'
)
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3'
]

# You also need to install the following packages, which are not available in
# pip.  They can both be installed by "git clone" and "python setup.py install",
# the normal way one installs a GitHub package.
#
# https://github.com/matplotlib/basemap
# https://github.com/tkrajina/srtm.py

PACKAGE_REQUIREMENTS = [
    'numpy',
    'scipy',
    'tensorflow',
    'keras',
    'scikit-learn',
    'scikit-image',
    'netCDF4',
    'pyproj',
    'opencv-python',
    'matplotlib',
    'pandas',
    'shapely',
    'descartes',
    'geopy',
    'metpy'
]

if __name__ == '__main__':
    setup(
        name='ml4rt',
        version='0.1',
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author='Ryan Lagerquist',
        author_email='ryan.lagerquist@noaa.gov',
        url='https://github.com/thunderhoser/ml4rt',
        packages=PACKAGE_NAMES,
        scripts=[],
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        include_package_data=True,
        zip_safe=False,
        install_requires=PACKAGE_REQUIREMENTS
    )
