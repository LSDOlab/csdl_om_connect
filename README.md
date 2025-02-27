# csdl_om_connect

[![GitHub Actions Test Badge](https://github.com/LSDOlab/csdl_om_connect/actions/workflows/install_test.yml/badge.svg)](https://github.com/LSDOlab/csdl_om_connect/actions)
[![License](https://img.shields.io/badge/License-GNU_LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

A package that interfaces models in CSDL with OpenMDAO and vice-versa

## Installation

To install the latest commit from the main branch, run the following command in the terminal:
```sh
pip install git+https://github.com/lsdolab/csdl_om_connect.git@main
```

To uninstall *csdl_om_connect*, run:
```sh
pip uninstall csdl_om_connect
```

To upgrade to the latest commit, uninstall *csdl_om_connect* and then reinstall it using:
```sh
pip uninstall csdl_om_connect
pip install git+https://github.com/lsdolab/csdl_om_connect.git@main
```

## Installation in development mode

To install *csdl_om_connect* in development mode, clone the repository and install it using:
```sh
git clone https://github.com/lsdolab/csdl_om_connect.git
pip install -e ./csdl_om_connect
```
The `-e` flag installs the package in editable mode, 
allowing you to modify the source code without needing to reinstall it.

To upgrade to the latest commit in development mode, navigate to the *csdl_om_connect* directory and run:
```sh
git pull
```

## Testing
To verify that the installed package works correctly, install `pytest` using:
```sh
pip install pytest
```

Then, run the following command from the project's root directory:
```sh
pytest
```

## Documentation
Given its role as an interface, *csdl_om_connect* does not have dedicated documentation.
Instead, we recommend users refer to the [examples](https://github.com/LSDOlab/csdl_om_connect/tree/main/examples) 
and docstrings within the code to get started and learn how to use the package.

## Bugs, feature requests, questions
Please use the [GitHub issue tracker](https://github.com/LSDOlab/csdl_om_connect/issues) 
for reporting bugs, requesting new features, or any other questions.

## License
This project is licensed under the terms of the [GNU Lesser General Public License v3.0](https://github.com/LSDOlab/csdl_om_connect/blob/main/LICENSE.txt).