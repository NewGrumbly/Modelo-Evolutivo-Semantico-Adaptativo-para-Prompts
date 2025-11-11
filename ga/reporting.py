# ga/reporting.py
import statistics
from typing import List, Dict, Any
from ga.genome import Individual

def get_fitness_stats(population: List[Individual]) -> Dict[str, Any]:
    """
    Calculates fitness statistics for a given population.
    """
    fitness_scores = [ind.get("fitness", 0.0) for ind in population]
    if not fitness_scores:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return {
        "count": len(fitness_scores),
        "mean": statistics.mean(fitness_scores),
        "std": statistics.pstdev(fitness_scores) if len(fitness_scores) > 1 else 0.0,
        "min": min(fitness_scores),
        "max": max(fitness_scores),
    }

def check_stagnation(
    fitness_history: List[float],
    stagnation_limit: int
) -> bool:
    """
    Checks if the average population fitness has stagnated (not improved)
    for a given number of generations.
    """
    if len(fitness_history) < stagnation_limit:
        return False # Not enough history to be stuck

    # Get the last limit number of fitness scores
    relevant_history = fitness_history[-stagnation_limit:]
    
    # Check if the fitness has not improved (is non-increasing)
    # We check if the max value in the recent history is the first value.
    # If the first value is still the max, it means we haven't improved.
    if relevant_history[0] >= max(relevant_history[1:]):
        return True
        
    return False