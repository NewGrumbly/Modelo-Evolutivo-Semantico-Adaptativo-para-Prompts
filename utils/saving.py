# utils/saving.py
import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from ga.genome import Individual

def save_population_to_json(population: List[Individual], file_path: Path):
    """
    Saves the list of individual dictionaries to a JSON file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(population, f, indent=2, ensure_ascii=False)

def append_metrics_to_csv(
    output_dir: Path,
    generation: int,
    stats: Dict[str, Any],
    duration_sec: float
):
    """
    Appends a row of statistics to the metrics_log.csv file.
    """
    csv_path = output_dir / "metrics_log.csv"
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writerow([
                "generation", "count", "mean_fitness", "std_fitness",
                "min_fitness", "max_fitness", "duration_sec"
            ])
        
        # Write data row
        writer.writerow([
            generation, stats["count"],
            f"{stats['mean']:.6f}", f"{stats['std']:.6f}",
            f"{stats['min']:.6f}", f"{stats['max']:.6f}",
            f"{duration_sec:.6f}"
        ])

def save_parameters_to_json(output_dir: Path, args: dict):
    """
    Save the execution arguments (parameters) to a JSON file
    inside the output directory.
    """
    params_path = output_dir / "parameters.json"
    
    # Convert arguments to a simple dictionary
    # (vars() works if args is an argparse object)
    try:
        params_data = vars(args).copy()
    except TypeError:
        # If 'args' is already a dictionary, we copy it
        params_data = args.copy()

    # Just remove the problematic key before saving.
    if 'outdir_base' in params_data:
        del params_data['outdir_base']
        
    try:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: parameters.json couldn't be saved: {e}")