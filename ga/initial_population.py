# ga/initial_population.py
import asyncio
from typing import List, Optional
from tqdm.asyncio import tqdm

from ga.genome import Individual
from agents.llm_agent import LLMAgent
from agents.role_agent import infer_role
from agents.synthesis_agent import generate_topic_and_prompt

# Default batch size to avoid overwhelming the LLM service
DEFAULT_BATCH_SIZE = 10

async def _create_one_individual(
    reference_text: str,
    llm_agent: LLMAgent,
    role_temp: float = 0.5,
    synthesis_temp: float = 0.7
) -> Optional[Individual]:
    """
    Private helper function to manage the sequential creation of a single individual.
    """
    try:
        # 1. Infer Role
        role = await infer_role(
            reference_text=reference_text,
            llm_agent=llm_agent,
            temperature=role_temp
        )
        if not role:
            # print("Warning: Failed to infer role. Skipping individual.")
            return None

        # 2. Generate Topic and Prompt (based on the inferred role)
        topic_prompt_tuple = await generate_topic_and_prompt(
            role=role,
            reference_text=reference_text,
            llm_agent=llm_agent,
            temperature=synthesis_temp
        )
        if not topic_prompt_tuple:
            # print("Warning: Failed to synthesize topic/prompt. Skipping individual.")
            return None
        
        topic, prompt = topic_prompt_tuple

        # 3. Construct the Individual object
        return Individual(
            role=role,
            topic=topic,
            prompt=prompt,
            generated_data=None,
            fitness=0.0 
        )

    except Exception as e:
        print(f"Error during individual creation: {e}")
        return None

async def create_initial_population(
    n: int,
    llm_agent: LLMAgent,
    reference_text: str
) -> List[Individual]:
    """
    Creates the initial population of n individuals.
    Uses a loop to ensure the successful creation of each individual.
    Processes individuals in parallel batches for improved efficiency.
    """
    print(f"🧬 Starting initial population generation for {n} individuals...")
    population: List[Individual] = []

    with tqdm(total=n, desc="Creating Gen 0", unit="ind") as pbar:
        while len(population) < n:
            # Determine how many individuals are needed
            n_needed = n - len(population) 

            # Define next batch size
            batch_size = min(n_needed, DEFAULT_BATCH_SIZE)

        # print(f"🔄 Generating batch of {batch_size} individuals... (Current: {len(population)}/{n})")

            # Create tasks for the batch
            tasks = [
                _create_one_individual(
                    reference_text=reference_text,
                    llm_agent=llm_agent
                ) for _ in range(batch_size)
            ]

            # Run tasks concurrently
            results = await asyncio.gather(*tasks)
        
            # Filter out any 'None' results from failed generations
            new_individuals = [ind for ind in results if ind is not None]

            # Add successfully created individuals to the population
            population.extend(new_individuals)

            # Update progress bar
            pbar.update(len(new_individuals))

            '''if new_individuals:
                print(f"✅ Batch complete. Added {len(new_individuals)} individuals.")
            else:
                print(f"⚠️ Batch complete. No individuals were added in this batch. Retrying...")'''
    
    print(f"✅ Initial population created. Generated {len(population)} individuals.")
    return population