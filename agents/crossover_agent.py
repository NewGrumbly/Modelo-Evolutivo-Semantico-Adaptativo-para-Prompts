# agents/crossover_agent.py
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from .llm_agent import LLMAgent
from ga.genome import Individual

# --- Pydantic Output Model ---
class CrossoverOutput(BaseModel):
    """
    Defines the JSON structure for the CrossoverAgent's output.
    """
    new_role: str = Field(
        ...,
        description="The resulting role for the child, either inherited from the strongest parent or a logical combination of both."
    )
    new_topic: str = Field(
        ...,
        description="The resulting topic for the child, either inherited or combined."
    )

# --- System Prompt ---
def _get_system_prompt() -> str:
    """
    Returns the system prompt for the Crossover Agent.
    This prompt instructs the LLM to perform a semantic crossover,
    choosing to either inherit or combine attributes.
    """
    return f"""
    You are an AI assistant specialized in genetic algorithms.
    Your task is to perform a semantic crossover between two parent individuals.
    You will receive the role and topic for each parent.

    Your goal is to generate a new and unique Role and Topic for a
    child individual that logically and creatively combines the
    characteristics of both parents.

    For each attribute (Role and Topic), you have two options:
    1.  Inherit: Choose the attribute from one parent if you deem it the
        strongest or most appropriate.
    2.  Combine: Create a new attribute that fuses the ideas from both
        parents (e.g., if Roles are "a healthcare worker" and "a concerned
        citizen", a combined Role could be "a healthcare worker describing
        their personal concerns").

    Your response MUST be a single JSON object conforming to the following schema:
    {CrossoverOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(parent1: Individual, parent2: Individual) -> str:
    """
    Returns the user prompt containing the genomes of the two parents.
    """
    return f"""
    Parent 1:
    - Role: "{parent1['role']}"
    - Topic: "{parent1['topic']}"

    Parent 2:
    - Role: "{parent2['role']}"
    - Topic: "{parent2['topic']}"

    Perform the semantic crossover to create the child's Role and Topic.
        """.strip()

# --- Main Agent Function ---
async def semantic_crossover(
    parent1: Individual,
    parent2: Individual,
    llm_agent: LLMAgent,
    temperature: float = 0.7
) -> Optional[Tuple[str, str]]:
    """
    Performs semantic crossover on two parent individuals.
    Returns a tuple (new_role, new_topic) or None on failure.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(parent1, parent2)

    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=CrossoverOutput,
        temperature=temperature
    )

    if isinstance(response_obj, CrossoverOutput):
        return response_obj.new_role, response_obj.new_topic
    
    return None