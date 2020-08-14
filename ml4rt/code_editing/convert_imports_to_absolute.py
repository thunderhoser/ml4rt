"""Converts all imports of ml4rt code from relative to absolute."""

import os
import glob


def _run():
    """Converts all imports of ml4rt code from relative to absolute.

    This is effectively the main method.
    """

    script_dir_name = os.getcwd()
    package_dir_name = '/'.join(script_dir_name.split('/')[:-1])

    module_file_pattern = '{0:s}/*.py'.format(package_dir_name)
    module_file_names = glob.glob(module_file_pattern)

    module_names = [
        os.path.splitext(os.path.split(f)[1])[0] for f in module_file_names
    ]

    for this_file_name in module_file_names:
        for this_module_name in module_names:
            this_command_string = (
                "sed -i 's/import {0:s}/from ml4rt import {0:s}/g' {1:s}"
            ).format(this_module_name, this_file_name)

            os.system(this_command_string)


if __name__ == '__main__':
    _run()
