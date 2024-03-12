# Makefile

## Variables
# SRC_DIR: Directory containing Python source files.
SRC_DIR = src
# FIGS_DIR: Directory for figures or any other generated files.
FIGS_DIR = figs
# DATA_DIR: Directory for figures or any other generated files.
DATA_DIR = data
# PYTHON: Python interpreter command.
PYTHON = python3
# Name of the Conda environment
CONDA_ENV_NAME = GMRI_env
# Path to the environment.yml file
ENVIRONMENT_FILE = GMRI_env.yml



# Command to create/activate the conda env
create_environment:
	conda env create -f $(ENVIRONMENT_FILE)

# create \figs, \srs, and \data directories
setup_dir:
	mkdir -p $(SRC_DIR) $(FIGS_DIR) $(DATA_DIR)
	@echo "Directories created: $(SRC_DIR) $(FIGS_DIR)"

# retrievs data and preps for eda
get_data: $(SRC_DIR) $(DATA_DIR) 
	@echo "Running file: src/prepare_data.py"
	python -B src/prepare_data.py
	@echo "Complete."

# runs eda, populates figures for EDA.md
run_eda: $(SRC_DIR) $(FIGS_DIR) $(DATA_DIR) data/combined_data_2006-2024.csv
	@echo "Running file: src/eda.py"
	python -B src/eda.py
	@echo "Complete."



clean:
	rm -rf data/*

