# Spontaneous behaviour is structured by reinforcement without explicit reward

## Authors
Jeffrey E. Markowitz<sup>*1,7</sup>, Winthrop Gillis<sup>*1</sup>, Maya Jay<sup>*1</sup>, Jeffrey Wood<sup>1</sup>, Ryley Harris<sup>1</sup>, Robert Cieszkowski<sup>1</sup>, Rebecca Scott<sup>1</sup>, David Brann<sup>1</sup>, Dorothy Koveal<sup>1</sup>, Tomasz Kula<sup>1</sup>, Caleb Weinreb<sup>1</sup>, Mohammed Abdal Monium Osman<sup>1</sup>, Sandra Romero Pinto<sup>2,3</sup>, Naoshige Uchida<sup>2,3</sup>, Scott W. Linderman<sup>4,5</sup>, Bernardo L. Sabatini<sup>1,6</sup>, Sandeep Robert Datta<sup>1,#</sup>

<br>

<sup>1</sup>Department of Neurobiology, Harvard Medical School, Boston, Massachusetts, United States<br>
<sup>2</sup>Department of Molecular and Cellular Biology, Harvard University, Cambridge, Massachusetts, United States<br>
<sup>3</sup>Center for Brain Science, Harvard University, Cambridge, Massachusetts, United States<br>
<sup>4</sup>Wu Tsai Neurosciences Institute, Stanford University, Stanford, California, United States<br>
<sup>5</sup>Department of Statistics, Stanford University, Stanford, California, United States<br>
<sup>6</sup>Howard Hughes Medical Institute, Chevy Chase, Maryland, United States<br>
<sup>7</sup>Present address: Wallace H. Coulter Department of Biomedical Engineering, Georgia Institute of Technology and Emory University. Atlanta, Georgia, United States<br>

#Corresponding Author 
*Co-first author

<br><br>

# Overview

Pre-processing and panel generation is done using Jupyter notebooks. You will need to have Jupyter or Jupyterlab installed on your machine. 

Much of the analysis was run on a Google Compute Engine VM with 128 CPUs and 850 GB of RAM. Long-running calculations were run on multiple machines using Slurm GCP https://github.com/FluidNumerics/slurm-gcp . 


<br><br>

# Installation

Installation instructions

1. (IF YOU DO NOT HAVE CONDA) Install Miniconda on your machine https://docs.conda.io/en/latest/miniconda.html . On linux you can use these commands in the terminal.

		# this example is for linux x86-64
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
		bash ~/miniconda.sh
1. We typically install jupyterlab in the base environment with `nb_conda_kernel`

		conda install -c conda-forge jupyterlab ipywidgets
		conda install nb_conda_kernels
		# be sure pip install ipykernel in other environments so they're detected by jupyter
1. Once Miniconda is installed create the environment used for nearly all notebooks in this repository by running from the command line.

		conda create -n spont-da python=3.10
		conda activate spont-da
1. Install the library needed to run the code by running `pip install -e .` while in the `spont-da` environment.
1. You will also want to install the environments specified in `hekcell.yaml`, `mousert-gpu.yaml` and `moseqdistance.yaml`. Notebooks with these files in square brackets indicate that you need to run the notebook in *that* environment. The top line in each file contains the command you need to install it.


<br><br>

# Data

How to obtain data.

1. Download data from Zenodo here **INSERT ZENODO BADGE POST-RELEASE**. You should see the following.

		.
		├── dlight_intermediate_results
		│   ├── lagged_analysis_session_bins.toml
		│   ├── syllable_stats_photometry_offline.toml
		│   └── syllable_stats_photometry_online.toml
		├── dlight_raw_data
		│   ├── dlight_photometry_processed_full.toml
		│   └── dlight_photometry_processed_full_transfer.parquet
		├── misc_intermediate_results
		│   └── autoencoder_characterization.parquet
		├── misc_raw_data
		│   ├── autoencoder_test_data.h5
		│   ├── f1_scores_estimates_actual_calls.parquet
		│   ├── latencies_stim.parquet
		│   ├── latencies_stim_arduino_test.dat
		│   └── spinograms.p
		├── optoda_intermediate_results
		│   ├── behavior_classes.toml
		│   ├── behavioral-distance.parquet
		│   ├── closed_loop_learners.toml
		│   ├── da-vs-learning-per-syllable.parquet
		│   ├── joint_syllable_map.toml
		│   ├── syllable_stats_offline.toml
		│   └── syllable_stats_online.toml
		├── optoda_raw_data
		│   ├── closed_loop_behavior_transfer.parquet
		│   ├── closed_loop_behavior_velocity_conditioned.parquet
		│   ├── closed_loop_behavior_with_simulated_triggers_transfer.parquet
		│   ├── learning_aggregate.parquet
		│   ├── learning_timecourse_binsize-30.parquet
		│   ├── learning_timecourse_processed.parquet
		│   └── learning_timecourse_processed_summary.parquet
		├── rl_intermediate_results
		│   ├── rl_model_heldout_results_best_lag_rands.p
		│   ├── rl_model_heldout_results_best_lag_rands.parquet
		│   ├── rl_model_heldout_results_lags.p
		│   ├── rl_model_heldout_results_lags.parquet
		│   ├── rl_model_parameters.toml
		│   └── rl_model_stats.toml
		└── rl_raw_data
			├── rl_modeling_dlight_data_offline.parquet
			└── rl_modeling_dlight_data_online.parquet
1. This contains everything you need to run the preprocessing and analysis notebooks.
1. Specific data formats were used to reduce file size for storage on Zenodo (brotli compression level 9). You will want to convert these files into formats that enable fast processing (snappy-compressed). This is done by running `_reformat_zenodo_downloads.ipynb`.

<br><br>

# Notebooks/scripts

1. First, open `analysis_configuration.toml` and alter the file paths in `[raw_data]` and `[intermediate_resluts]` to reflect where you downloaded the data. `[figures.store_dir]` is where panels will be saved when generated.
1. Next, you will need to run all preprocessing notebooks found in `notebooks_preprocessing`.
	1. (If you skipped this step above) run `_reformat_zenodo_downloads.ipynb`.
	1. Start with `dlight_00-14`. Note you can skip `13-14`, the results of the RL model grid search and fit are included in this dataset. You can recompute if you like.
	1. Next run `behavior_00-03`.
1. Note that some notebooks will need to be run multiple times with different settings (e.g. to perform analysis from separate datasets or with different settings). This is mentioned in each notebook where relevant.
1. With all of the intermediate data generated you can generate individual panels by running notebooks in `notebooks_panels`.

<br><br><br>

# Troubleshooting

1. If you get a `no space left on device error` during a long-running multi-processing calculation via Joblib this is likely due to `/dev/shm` filling up on a Linux system. A simple fix is to set `import tempfile; temp_folder=tempfile.gettempdir()` in your `Parallel` call. This can also be set via the `JOBLIB_TEMP_FOLDER` environment variable. In the bash environment you are running Jupyter from set `export JOBLIB_TEMP_FOLDER=/tmp`. Or in your notebook `import os; os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp' `
2. If your kernel dies while attempting to run `_reformat_zenodo_downloads.ipynb`, make sure you're running the notebook on a machine with at least 160GB of memory, as one of the data files (`optoda_raw_data/closed_loop_behavior_transfer.parquet`) uses that much memory to load. This preprocessing step reduces the amount of memory you need to run subsequent figure generating notebooks.
