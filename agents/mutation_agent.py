# agents/mutation_agent.py
from pydantic import BaseModel, Field
from typing import Optional, Tuple
import random
from .llm_agent import LLMAgent
from ga.genome import Individual

# --- Pydantic Output Model ---
class MutationOutput(BaseModel):
    """
    Defines the JSON structure for the MutationAgent's output.
    It returns a single new value for the attribute being mutated.
    """
    new_value: str = Field(
        ...,
        description="The new, mutated value (either a role or a topic) for the attribute."
    )

# --- System Prompts ---
def _get_system_prompt_refine() -> str:
    """
    Returns the system prompt for Mode 1: Re-conceptualization (Refinement).
    This mode explores the local neighborhood of the solution.
    """
    return f"""
    You are an AI assistant specialized in genetic algorithms. Your
    task is to perform a semantic mutation via Re-conceptualization.

    Your job is to suggest a slightly different, improved version of the
    given text. The new version must keep the main intent and must
    remain relevant to the provided context.

    Your response MUST be a single JSON object conforming to the following schema:
    {MutationOutput.model_json_schema()}
        """.strip()

def _get_system_prompt_explore() -> str:
    """
    Returns the system prompt for Mode 2: Creative Leap (Exploration).
    This mode jumps to a new area of the solution space to escape
    local optima when the population is stuck.
    """
    return f"""
    You are an AI assistant specialized in genetic algorithms. Your
    task is to perform a semantic mutation via Creative Leap.

    Your job is to suggest a significantly different and creative
    alternative for the given text. The new version must explore a new
    concept, but must still be relevant to the provided context.

    Your response MUST be a single JSON object conforming to the following schema:
    {MutationOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(
    reference_text: str,
    attribute_to_mutate: str,
    value_to_mutate: str,
    context_attribute: str,
    context_value: str
) -> str:
    """
    Returns the user prompt for mutation.
    It includes the reference text and the other attribute
    as contextual anchors to keep the mutation relevant.
    """
    return f"""
    Context:
    - Reference Text: "{reference_text}"
    - (Context Anchor) {context_attribute}: "{context_value}"

    Task:
    Mutate this {attribute_to_mutate}: "{value_to_mutate}"
        """.strip()

# --- Main Agent Function ---
async def semantic_mutation(
    individual: Individual,
    reference_text: str,
    llm_agent: LLMAgent,
    is_stuck: bool,
    temperature: float = 0.8
) -> Optional[Tuple[str, str]]:
    """
    Performs semantic mutation on an individual.
    Returns a tuple (new_role, new_topic) or None on failure.
    """
    
    # 1. Decide which attribute to mutate
    if random.random() < 0.5:
        attribute_to_mutate = "role"
        value_to_mutate = individual['role']
        context_attribute = "topic"
        context_value = individual['topic']
    else:
        attribute_to_mutate = "topic"
        value_to_mutate = individual['topic']
        context_attribute = "role"
        context_value = individual['role']

    # 2. Select the system prompt based on the 'is_stuck' flag
    if is_stuck:
        system_prompt = _get_system_prompt_explore()
    else:
        system_prompt = _get_system_prompt_refine()
        
    # 3. Create the user prompt
    user_prompt = _get_user_prompt(
        reference_text=reference_text,
        attribute_to_mutate=attribute_to_mutate,
        value_to_mutate=value_to_mutate,
        context_attribute=context_attribute,
        context_value=context_value
    )

    # 4. Call the LLM
    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=MutationOutput,
        temperature=temperature
    )

    if not isinstance(response_obj, MutationOutput):
        return None # LLM call failed

    # 5. Reconstruct the genome
    new_value = response_obj.new_value
    if attribute_to_mutate == "role":
        return new_value, individual['topic'] # (new_role, old_topic)
    else:
        return individual['role'], new_value # (old_role, new_topic)