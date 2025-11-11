# ga/evolution.py
import random
import asyncio
import time
from typing import List, Optional

from ga.genome import Individual
from agents.llm_agent import LLMAgent
from metrics.fitness import evaluate_population_fitness
from ga.reporting import get_fitness_stats, check_stagnation
from utils.saving import append_metrics_to_csv

# Import Semantic Operators
from agents.crossover_agent import semantic_crossover
from agents.mutation_agent import semantic_mutation
from agents.regenerate_prompt_agent import regenerate_prompt
from agents.generate_data_agent import generate_data_for_individual

# --- Constants ---
CHILD_BATCH_SIZE = 10   # Max parallel LLM calls for new children
STAGNATION_LIMIT = 3    # Generations to wait before enabling "Creative Leap"

# --- Selection ---
def tournament_selection(population: List[Individual], k: int = 3) -> Individual:
    """Selects an individual using k-tournament selection."""
    candidates = random.sample(population, k)
    return max(candidates, key=lambda ind: ind["fitness"])

# --- Child Pipeline ---
async def _process_child_pipeline(
    parent1: Individual,
    parent2: Individual,
    reference_text: str,
    llm_agent: LLMAgent,
    prob_crossover: float,
    prob_mutation: float,
    is_stuck: bool
) -> Optional[Individual]:
    """
    The full asynchronous pipeline to create one new child.
    Implements the sequential probability model.
    """
    try:
        new_role, new_topic = None, None
        
        # 1. Crossover or Reproduction (Copy)
        if random.random() < prob_crossover:
            # Crossover
            result = await semantic_crossover(parent1, parent2, llm_agent)
            if result: new_role, new_topic = result
        else:
            # Reproduction (Copy parent 1)
            new_role, new_topic = parent1['role'], parent1['topic']
            
        if not (new_role and new_topic):
            return None # Crossover failed or parent was invalid

        # 2. Mutation (Independent)
        if random.random() < prob_mutation:
            # We mutate the genome after it has been crossed or copied
            temp_individual = Individual(role=new_role, topic=new_topic, prompt="", generated_data=None, fitness=0.0)
            result = await semantic_mutation(temp_individual, reference_text, llm_agent, is_stuck)
            if result: new_role, new_topic = result
            
            if not (new_role and new_topic):
                return None # Mutation failed
        
        # 3. Regenerate Prompt
        new_prompt = await regenerate_prompt(new_role, new_topic, reference_text, llm_agent)
        if not new_prompt:
            return None # Prompt regeneration failed

        # 4. Build the individual (unevaluated)
        child = Individual(
            role=new_role,
            topic=new_topic,
            prompt=new_prompt,
            generated_data=None,
            fitness=0.0
        )

        # 5. Generate Data
        new_data = await generate_data_for_individual(child, reference_text, llm_agent)
        if not new_data:
            return None # Data generation failed

        child['generated_data'] = new_data
        return child # Return the complete, unevaluated child
        
    except Exception:
        return None # Fail-safe

# --- Main Evolution Loop ---
async def run_evolution(
    population: List[Individual],
    reference_text: str,
    llm_agent: LLMAgent,
    bert_model: str,
    generations: int,
    k_tournament: int,
    prob_crossover: float,
    prob_mutation: float,
    elite_size: int,
    output_dir: Path
) -> List[Individual]:
    """
    The main asynchronous GA loop.
    """
    print("\n--- 🚀 Starting Evolution ---")
    current_population = population
    pop_size = len(current_population)
    fitness_history = [get_fitness_stats(current_population)["mean"]]
    is_stuck = False

    for g in range(1, generations + 1):
        gen_start_time = time.time()
        print(f"\n--- Generation {g}/{generations} ---")
        
        # Sort population by fitness (descending)
        current_population.sort(key=lambda ind: ind["fitness"], reverse=True)
        
        # 1. Check for Stagnation
        is_stuck = check_stagnation(fitness_history, STAGNATION_LIMIT)
        if is_stuck:
            print("   ⚠️ Stagnation detected! Enabling 'Creative Leap' mutations.")

        # 2. Elitism
        # The best individuals are copied directly to the next generation
        new_population = current_population[:elite_size]
        print(f"   Elite size: {len(new_population)} individuals preserved.")
        
        # 3. Create Children (in parallel batches)
        children_to_create = pop_size - elite_size
        new_children: List[Individual] = []
        
        while len(new_children) < children_to_create:
            n_needed = children_to_create - len(new_children)
            batch_size = min(n_needed, CHILD_BATCH_SIZE)
            print(f"   ... Launching child batch of {batch_size} (Target: {len(new_children)}/{children_to_create})")

            tasks = []
            for _ in range(batch_size):
                p1 = tournament_selection(current_population, k=k_tournament)
                p2 = tournament_selection(current_population, k=k_tournament)
                tasks.append(
                    _process_child_pipeline(
                        p1, p2, reference_text, llm_agent,
                        prob_crossover, prob_mutation, is_stuck
                    )
                )
            
            # Run tasks in parallel
            results = await asyncio.gather(*tasks)
            
            successful_children = [child for child in results if child is not None]
            new_children.extend(successful_children)
            print(f"   ... Batch complete. {len(successful_children)} successful children.")

        # 4. Evaluate all new children in one batch
        print(f"   ... Evaluating fitness for {len(new_children)} new children...")
        evaluated_children = evaluate_population_fitness(
            population=new_children,
            reference_text=reference_text,
            generation=g,
            max_generations=generations,
            bert_model=bert_model
        )
        
        # 5. Form the final new population
        new_population.extend(evaluated_children)
        current_population = new_population

        # 6. Report stats for the new generation
        stats = get_fitness_stats(current_population)
        fitness_history.append(stats["mean"])
        gen_time = time.time() - gen_start_time

        # Save metrics to CSV
        append_metrics_to_csv(
            output_dir=output_dir,
            generation=g,
            stats=stats,
            duration_sec=gen_time
        )

        print(f"   Generation {g} complete. Time: {gen_time:.2f}s")
        print(f"   Stats: Avg Fitness: {stats['mean']:.4f} | Max Fitness: {stats['max']:.4f}")

    print("--- ✅ Evolution Finished ---")
    return current_population