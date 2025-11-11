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