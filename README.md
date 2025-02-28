# EMSL-XCT

This repository contains the codebase for the EMSL-XCT autmation project, which contains several Python-based XRCT reconstruction algorithms. The project is designed to provide a flexible and efficient workflow for doing XCT image reconstructions, with a focus on ease of use and extensibility. The code is organized into several modules, each of which implements a different reconstruction algorithm or utility function.

## Set up
To set up EMSL-XCT environment, conda-forge channe is used where available, for others pip was used. First, clone the repository to your local machine:

Then you can make a new conda environment and install the requirements file. {# Will make a shell script to clean out the installation process later}

Using the Conda-Forge channel for most of the packages, and pip for the mbirjax package.

```bash
conda create -n emsl-xct python=3.8
conda activate emsl-xct
conda install --file conda-requirements.txt -c conda-forge
pip install -r pip-requirements.txt
```
Then add the IPython kernel to Jupyter notebook:

```bash
python -m ipykernel install --user --name=emsl-xct --display-name=emsl-xct"

```

Then you can select the ```emsl-xct``` kernel in Jupyter notebook.

## Usage

The main script that contains the file selection, Centre of Rotation (CoR) calculation, and reconstruction is the ```recon.ipynb``` notebook. The supporting functions are stored in the python scripts in the Backend folder. The ```recon.ipynb``` notebook is designed to be user-friendly and easy to use. The user can select the reconstruction algorithm (currently it only has the simple FDR and MBIRJAX methods), the data file, and the parameters for the reconstruction and ring removal methods. The notebook will then run the reconstructions and display the slices.

For the post processing steps those images can be stored as a image stack that can be processed by Avizo or ImageJ.

### Selecting Data Folder

After importing the necessary packages the user can select the Data folder which contains the projection images and the metadata files (i.e. .ang, .xml files) for the reconstruction. When you slect the data folder the scanning parameters are read by the support functions in the backend folder. 

### Finding the Centre of Rotation (CoR)

Once the user selects the Data folder they can auto calculate the CoR by selecting the first and last projection images. The user can also manually input the CoR using the slider bar on the widget. If the auto calculation is not accurate the user can manually adjust the CoR using the slider. The auto calulation is done by finding the maximum of the cross-correlation between the first and last projection images.

### Reconstruction




