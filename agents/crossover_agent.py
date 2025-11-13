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
    Your task is to perform a semantic crossover between two parent individuals
    based on their relevance to a Reference Text.

    You will receive the Role and Topic for each parent, and the Reference Text.
    Your goal is to generate a new Role and Topic for a child that is 
    highly coherent with the Reference Text.

    For each attribute (Role and Topic), you have two options:
    1.  Inherit: Analyze both parent attributes and the Reference Text.
        Choose the attribute (from Parent 1 or 2) that is semantically stronger
        and more relevant to the Reference Text.
    2.  Combine: If both attributes are strong and relevant, create a
        new attribute that fuses their ideas in a way that is still
        perfectly aligned with the Reference Text.

    Your response MUST be a single JSON object conforming to the following schema:
    {CrossoverOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(parent1: Individual, parent2: Individual, reference_text:str) -> str:
    """
    Returns the user prompt containing the genomes of the two parents
    and Reference Text.
    """
    return f"""
    Reference Text (Your anchor for all decisions):
    "{reference_text}"

    Parent 1:
    - Role: "{parent1['role']}"
    - Topic: "{parent1['topic']}"

    Parent 2:
    - Role: "{parent2['role']}"
    - Topic: "{parent2['topic']}"

    Perform the semantic crossover based on relevance to the Reference Text.
        """.strip()

# --- Main Agent Function ---
async def semantic_crossover(
    parent1: Individual,
    parent2: Individual,
    reference_text: str,
    llm_agent: LLMAgent,
    temperature: float = 0.7
) -> Optional[Tuple[str, str]]:
    """
    Performs semantic crossover on two parent individuals.
    Returns a tuple (new_role, new_topic) or None on failure.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(parent1, parent2, reference_text)

    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=CrossoverOutput,
        temperature=temperature
    )

    if isinstance(response_obj, CrossoverOutput):
        return response_obj.new_role, response_obj.new_topic
    
    return None