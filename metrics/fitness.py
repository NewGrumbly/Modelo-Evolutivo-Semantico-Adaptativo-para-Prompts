# metrics/fitness.py
from typing import List
from bert_score import score as bert_score
from ga.genome import Individual
from metrics.diversity import calculate_compression_ratio, calculate_internal_repetition

# --- Fitness Function Configuration ---

# 1. Coherence Penalties
# We penalize individuals that are *too* similar to the reference.
COHERENCE_UPPER_THRESHOLD = 0.8
# How strongly to penalize (50% penalty on the amount exceeded)
STATIC_COHERENCE_PENALTY_FACTOR = 0.5 

# 2. Diversity Penalties
# We penalize individuals with low internal diversity.
# We set thresholds for "what is considered bad".
COMPRESSION_THRESHOLD = 2.0 # Higher is worse (more redundant)
REPETITION_THRESHOLD = 0.5  # Higher is worse (20% internal repetition)
# How strongly to penalize for low diversity
DYNAMIC_DIVERSITY_PENALTY_FACTOR = 0.1
# -------------------------------------

def _apply_penalties(
    individual: Individual,
    generation: int,
    max_generations: int
) -> Individual:
    """
    Applies static and dynamic penalties to an individual's base fitness score.
    """
    base_coherence = individual['fitness'] # This is the raw BERTScore
    text = individual['generated_data'] or ""
    total_penalty = 0.0

    # 1. Static Penalty: Penalize for Excessive Coherence
    if base_coherence > COHERENCE_UPPER_THRESHOLD:
        # Calculate how much it exceeded the threshold
        exceeded_amount = base_coherence - COHERENCE_UPPER_THRESHOLD
        penalty = exceeded_amount * STATIC_COHERENCE_PENALTY_FACTOR
        total_penalty += penalty

    # 2. Dynamic Penalty: Penalize for Low Diversity
    # Calculate diversity metrics
    compression_ratio = calculate_compression_ratio(text)
    repetition_rate = calculate_internal_repetition(text)

    # Determine if the individual has low diversity
    is_low_diversity = (
        compression_ratio > COMPRESSION_THRESHOLD or
        repetition_rate > REPETITION_THRESHOLD
    )

    if is_low_diversity:
        # Penalty gets stronger as generations go on
        dynamic_factor = (generation / max_generations)
        penalty = DYNAMIC_DIVERSITY_PENALTY_FACTOR * dynamic_factor
        total_penalty += penalty

    # Apply final penalties and clamp fitness (it can't be negative)
    individual['fitness'] = max(0.0, base_coherence - total_penalty)
    return individual

def evaluate_population_fitness(
    population: List[Individual],
    reference_text: str,
    generation: int,
    max_generations: int,
    bert_model: str = "bert-base-uncased"
) -> List[Individual]:
    """
    Calculates the fitness for an entire population.
    
    1. Runs a batch BERTScore for base coherence.
    2. Applies individual penalties (static and dynamic) to each member.
    """
    # Step 1: Batch Coherence (BERTScore)
    # Get all candidate texts and a reference text
    candidates = [ind['generated_data'] or "" for ind in population]
    references = [reference_text] * len(population)
    
    if candidates:
        # Run BERTScore once for all individuals
        _, _, F1 = bert_score(
            cands=candidates,
            refs=references,
            model_type=bert_model,
            lang="en",
            verbose=False
        )
        f1_scores = F1.tolist()
        
        # Assign the base coherence (F1 score) to each individual
        for i, ind in enumerate(population):
            ind['fitness'] = f1_scores[i]
    
    # Step 2: Apply Individual Penalties
    # Iterate and apply the complex fitness logic to each one
    evaluated_population = [
        _apply_penalties(ind, generation, max_generations)
        for ind in population
    ]
    
    return evaluated_population