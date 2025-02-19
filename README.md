# EMSL-XCT

This repository contains the codebase for the EMSL-XCT autmation project, which contains several Python-based XRCT reconstruction algorithms. The project is designed to provide a flexible and efficient workflow for doing XCT image reconstructions, with a focus on ease of use and extensibility. The code is organized into several modules, each of which implements a different reconstruction algorithm or utility function.

## Installation
To install the EMSL-XCT package, conda is recommended but pip can also be used with modofocations. First, clone the repository to your local machine:

Then you can make a new conda environment and install the requirements file. {# Will make a shell script to clean out the installation process later}

```bash
conda create -n emsl-xct python=3.8
conda activate emsl-xct
conda install --file conda-requirements.txt -c conda-forge
pip install -r pip-requirements.txt
```


