#!/bin/bash
# run_experiment.sh

# This script prepares the Conda environment and PATH,
# then runs irace. It is designed to be called by nohup.

# 1. Find the path to Conda and set up the environment
source /home/colossus/miniconda3/bin/activate irace_env

# 2. Add the irace 'bin' directory to the PATH
# This finds the 'irace' executable inside the R library
export PATH="$(Rscript -e "cat(system.file(package='irace', 'bin', mustWork=TRUE))"):$PATH"

# 3. Run iRace 
irace --scenario scenario.txt