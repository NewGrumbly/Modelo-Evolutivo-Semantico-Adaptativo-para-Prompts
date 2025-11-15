#!/bin/bash
# run_experiment.sh

# This script prepares the Conda environment and PATH,
# then runs irace. It is designed to be called by nohup.

# 1. Find the path to Conda and set up the environment
source /home/colossus/miniconda3/bin/activate irace_env

# 2. Run iRace 
irace --scenario scenario.txt > irace_console.log 2>&1