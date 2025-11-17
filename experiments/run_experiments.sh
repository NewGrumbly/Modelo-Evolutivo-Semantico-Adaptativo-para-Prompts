#!/bin/bash
# run_experiments.sh

# This script runs the 9 FINAL experiment jobs.
# It uses ONE fixed configuration, tested 3 times on 3 reference texts.
# It runs sequentially (one job after another) for maximum stability.
#
# Total Jobs: 9 (1 Config x 3 Texts x 3 Repetitions)
# Estimated Runtime (Sequential): ~6.2 Days (9 jobs * ~16.5h/job)

# Exit immediately if a command fails
set -e

# --- 1. CONFIGURATION ---
# Absolute path to your venv's Python executable
PYTHON_VENV_PATH="../venv/bin/python" 

# Path to your main.py script
MAIN_SCRIPT_PATH="../main.py"
# ----------------------------------------


# --- 2. TASK DEFINITIONS ---
# Paths to reference texts
TEXTS=(
    "reference_texts/text_01.txt"
    "reference_texts/text_02.txt"
    "reference_texts/text_03.txt"
)

# Number of repetitions per text
REPETITIONS=(1 2 3)

# The single, fixed output directory for all 9 runs
# Your main.py will create timestamped subfolders inside this.
OUT_DIR_BASE="exec/final_runs"
mkdir -p "$OUT_DIR_BASE" # Create it just in case

# The FINAL, FIXED parameters from your preliminary proposal
FIXED_PARAMS="--n 100 --generations 100 --k 5 --elite_size 5 --prob_crossover 0.8 --prob_mutation 0.05"

# --- 3. HELPER FUNCTIONS ---
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S"
}

# Simple function to run one job and log its start/end
run_job() {
  local job_name="$1"
  local full_command="$2"

  echo ""
  echo "-------------------------------------------------"
  echo "--- [$(timestamp)] STARTING JOB: $job_name"
  echo "--- Command: $full_command"
  echo "-------------------------------------------------"
  
  # Execute the command.
  # 'set -e' (at top) will make the script stop if this fails.
  $full_command

  echo "--- [$(timestamp)] JOB SUCCEEDED: $job_name"
}

# --- 4. MAIN EXECUTION ---
main() {
  echo "================================================="
  echo "== STARTING FINAL EXPERIMENT (9 JOBS)"
  echo "== Start Time: $(timestamp)"
  echo "================================================="

  # Loop through each text
  for text_file in "${TEXTS[@]}"; do
    
    # Get a short name for the text (e.g., "text_01")
    local text_name=$(basename $text_file .txt)
    echo ""
    echo "--- Processing Text: $text_file ---"

    # Loop through each repetition for this text
    for rep_num in "${REPETITIONS[@]}"; do
      
      local job_name="${text_name}_Rep${rep_num}"
      
      run_job "$job_name" \
              "$PYTHON_VENV_PATH $MAIN_SCRIPT_PATH $FIXED_PARAMS --reference_text $text_file --outdir_base $OUT_DIR_BASE"
    done
  done

  echo ""
  echo "================================================="
  echo "== ALL 9 JOBS COMPLETE."
  echo "== End Time: $(timestamp)"
  echo "================================================="
}

# Execute the main function
main