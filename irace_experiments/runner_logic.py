# target_runner.py
# This script acts as a bridge between irace and main.py
# It is designed to be executed from WITHIN the 'irace_experiment/' directory.

import sys
import subprocess
import pathlib
import csv
import re
import argparse
import math

# --- Configuration ---
# Path to the main Python script, relative to this runner
MAIN_SCRIPT_PATH = "../main.py" 
# Path to the output directory that main.py creates
EXEC_DIR_PATH = "../exec" 
# ---------------------

def get_latest_exec_dir(base_dir=EXEC_DIR_PATH):
    """
    Finds the most recent experiment directory created by main.py.
    Based on the timestamp format from your utils/setup.py.
    """
    base_path = pathlib.Path(base_dir)
    if not base_path.exists():
        return None
    
    # Filter by the timestamp format your setup script uses
    dirs = [d for d in base_path.iterdir() if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", d.name)]
    if not dirs:
        return None
        
    return max(dirs, key=lambda d: d.stat().st_mtime)

def get_final_max_fitness(exec_dir):
    """
    Reads the metrics_log.csv and gets the 'max_fitness' from the last generation.
    """
    metrics_file = exec_dir / "metrics_log.csv"
    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found.", file=sys.stderr)
        return None

    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            last_line = None
            for line in reader:
                last_line = line
            
            if last_line:
                # Find the 'max_fitness' column index
                max_fitness_index = header.index("max_fitness")
                return float(last_line[max_fitness_index])
                
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        return None
    return None

def main():
    # 1. Arguments passed from irace:
    # sys.argv[1] = config_id
    # sys.argv[2] = instance_id
    # sys.argv[3] = seed
    # sys.argv[4] = instance_path (e.g., 'reference_texts/text_01.txt')
    # sys.argv[5:] = the parameters (e.g., '--n 100', '--prob_mutation 0.01', ...)
    
    instance_path = sys.argv[4]
    irace_params = sys.argv[5:]

    # 2. Use argparse to parse the parameters from irace
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--generations", type=int, required=True)
    parser.add_argument("--prob_mutation", type=float, required=True)
    parser.add_argument("--prob_crossover", type=float, required=True)
    parser.add_argument("--k_perc", type=float, required=True) # %
    parser.add_argument("--elit_perc", type=float, required=True) # %
    
    args = parser.parse_args(irace_params)

    # 3. LOGIC: Translate percentages to integers for main.py
    # Your main.py expects integers for k and elite_size
    #
    
    # k (tournament size)
    k_int = max(3, math.ceil(args.k_perc * args.n)) # At least 3 for a tournament
    
    # elite_size
    elit_int = max(1, math.ceil(args.elit_perc * args.n)) # At least 1 elite
    
    # 4. Build the final command to call your main.py
    python_exe = "/home/colossus/LLM/grumbly/Modelo-Evolutivo-Semantico-Adaptativo-para-Prompts/venv/bin/python"
    command = [
        python_exe,
        MAIN_SCRIPT_PATH,
        "--n", str(args.n),
        "--generations", str(args.generations),
        "--prob_mutation", str(args.prob_mutation),
        "--prob_crossover", str(args.prob_crossover),
        "--k", str(k_int),
        "--elite_size", str(elit_int),
        "--reference_text", instance_path,
        "--outdir_base", EXEC_DIR_PATH # Tell main.py where to save results
    ]
    
    # 5. Execute main.py
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print("Error: main.py failed. See stderr below:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1) # Tell irace this run failed

    # 6. Find the result directory
    latest_dir = get_latest_exec_dir()
    if not latest_dir:
        print("Error: Could not find result directory in {EXEC_DIR_PATH}.", file=sys.stderr)
        sys.exit(1)
        
    # 7. Read the final fitness
    final_fitness = get_final_max_fitness(latest_dir)
    if final_fitness is None:
        print(f"Error: Could not read fitness from {latest_dir}.", file=sys.stderr)
        sys.exit(1)
        
    # 8. Return the COST to irace
    # irace MINIMIZES. We want to MAXIMIZE fitness.
    # So, Cost = 1.0 - Fitness
    cost = 1.0 - final_fitness
    
    # This is the only line irace reads as the result
    print(f"{cost:.6f}")

if __name__ == "__main__":
    main()