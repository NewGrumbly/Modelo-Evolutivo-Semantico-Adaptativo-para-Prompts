# main.py
import argparse
import asyncio
import time
from pathlib import Path

# Utilities
from utils.setup import setup_experiment
from utils.saving import save_population_to_json
from ga.reporting import get_fitness_stats # Para el reporte de Gen 0

# Pipeline Modules
from agents.llm_agent import LLMAgent
from ga.initial_population import create_initial_population
from metrics.fitness import evaluate_population_fitness
from ga.evolution import run_evolution, CHILD_BATCH_SIZE
from agents.generate_data_agent import generate_data_for_individual
from utils.saving import append_metrics_to_csv

async def main():
    parser = argparse.ArgumentParser(description="Evolutionary Prompt Generation")

    # --- GA Parameters ---
    parser.add_argument("--n", type=int, default=30, help="Population size.")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations.")
    parser.add_argument("--k", type=int, default=3, help="Tournament size for selection.")
    parser.add_argument("--elite_size", type=int, default=2, help="Number of elite individuals to preserve.")
    
    # -- Genetic Operator Probabilities ---
    parser.add_argument("--prob_crossover", type=float, default=0.8, help="Probability of crossover.")
    parser.add_argument("--prob_mutation", type=float, default=0.1, help="Probability of mutation.")

    # --- Model Parameters ---
    parser.add_argument("--model", default="llama3", help="Ollama LLM model to use.")
    parser.add_argument("--bert_model", default="bert-base-uncased", help="BERT model for fitness evaluation.")

    # --- IO Parameters ---
    parser.add_argument("--outdir_base", type=Path, default=Path("exec"), help="Base directory for experiment output.")
    parser.add_argument("--reference_text", type=str, default=None, help="Specific reference text file to use (optional).")
    
    args = parser.parse_args()

    # --- 1. Setup Experiment ---
    print("--- 1/5: Setting up experiment directory ---")
    output_dir, reference_text = setup_experiment(
        base_dir=args.outdir_base,
        reference_text_arg=args.reference_text
    )
    print(f"   → Output will be saved to: {output_dir}")
    print(f"   → Reference text loaded.")

    total_start_time = time.time()
    
    # --- 2. Initialize LLM Agent ---
    llm_agent = LLMAgent(model=args.model)

    # --- 3. Initial Population (Generation 0) ---
    print(f"\n--- 2/5: Creating Initial Population (n={args.n}) ---")
    # This step creates N individuals, but 'generated_data' is None
    population_gen0 = await create_initial_population(
        n=args.n,
        llm_agent=llm_agent,
        reference_text=reference_text
    )
    
    # --- 4. Generate Data & Evaluate Gen 0 ---
    print("\n--- 3/5: Generating data and evaluating Gen 0 ---")
    gen0_start_time = time.time()

    # We must ensure all N individuals have data before evaluating.
    # We find all individuals that need data generated.
    individuals_to_generate = [ind for ind in population_gen0 if ind['generated_data'] is None]
    
    # We process them in batches until all have data
    while individuals_to_generate:
        n_needed = len(individuals_to_generate)
        batch_size = min(n_needed, CHILD_BATCH_SIZE)
       # print(f"   ... Generating data for batch of {batch_size} (Remaining: {n_needed})")

        # Create tasks for the current batch
        tasks = [
            generate_data_for_individual(ind, reference_text, llm_agent)
            for ind in individuals_to_generate[:batch_size]
        ]
        
        results = await asyncio.gather(*tasks)
        
        # We process the results of this batch
        successful_indices = []
        for i, data in enumerate(results):
            if data:
                # If successful, assign the data back to the individual
                individuals_to_generate[i]['generated_data'] = data
                successful_indices.append(i)
        
        # Update the list of individuals that still need data
        # We rebuild the list by removing those that were successful
        individuals_to_generate = [
            ind for i, ind in enumerate(individuals_to_generate)
            if i not in successful_indices
        ]

    print("   ... All individuals for Gen 0 have generated data.")
    
    # Now that all 'generated_data' fields are filled, evaluate fitness of Gen 0
    evaluated_pop_gen0 = evaluate_population_fitness(
        population=population_gen0,
        reference_text=reference_text,
        generation=0,
        max_generations=args.generations,
        bert_model=args.bert_model
    )
    
    # Now we get the total time for Gen 0 (Data Gen + Eval)
    gen0_total_time = time.time() - gen0_start_time
    
    # Save Gen 0 metrics
    gen0_stats = get_fitness_stats(evaluated_pop_gen0)
    append_metrics_to_csv(output_dir, 0, gen0_stats, gen0_total_time)
    save_population_to_json(evaluated_pop_gen0, output_dir / "population_gen_0.json")
    
    print(f"   → Gen 0 evaluated. Avg Fitness: {gen0_stats['mean']:.4f}")
    
    # --- 5. Run Evolution ---
    print(f"\n--- 4/5: Starting evolution for {args.generations} generations ---")
    
    final_population = await run_evolution(
        population=evaluated_pop_gen0,
        reference_text=reference_text,
        llm_agent=llm_agent,
        bert_model=args.bert_model,
        generations=args.generations,
        k_tournament=args.k,
        prob_crossover=args.prob_crossover,
        prob_mutation=args.prob_mutation,
        elite_size=args.elite_size,
        output_dir=output_dir
    )
    
    # --- 6. Final Consolidated Evaluation ---
    print("\n--- 5/5: Running Final Consolidated Evaluation ---")
    # Re-evaluate the entire final population using the final generation's
    # dynamic penalty, ensuring all individuals are scored by the same standard.
    final_evaluated_population = evaluate_population_fitness(
        population=final_population,
        reference_text=reference_text,
        generation=args.generations,
        max_generations=args.generations,
        bert_model=args.bert_model
    )
    
    # Save the final, re-scored population
    save_population_to_json(final_evaluated_population, output_dir / "population_final.json")
    
    total_time = time.time() - total_start_time
    print(f"\n--- ✅ Process Complete ---")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Final results saved in: {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())