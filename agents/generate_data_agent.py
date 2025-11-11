# agents/generate_data_agent.py
from pydantic import BaseModel, Field
from typing import Optional
from .llm_agent import LLMAgent
from ga.genome import Individual

# --- Pydantic Output Model ---
class DataOutput(BaseModel):
    """
    Defines the expected JSON structure for the GenerateDataAgent's output.
    """
    generated_text: str = Field(
        ...,
        description="The single, high-quality, short output text that fulfills the instruction."
    )

# --- System Prompt ---
def _get_system_prompt() -> str:
    """
    Returns the system prompt for the Generate Data Agent.
    This prompt instructs the LLM to act as the final text generator.
    """
    return f"""
    You are an AI text generator. Your task is to generate a single,
    high-quality, concise output (1-2 sentences maximum) that fulfills the instruction given in
    the Prompt.

    Your response must be strictly aligned with the Role, Topic, and
    Reference Text provided as context.

    CRITICAL RULES:
    1.  The output text MUST be text-only.
    2.  The output text MUST be short (1-2 sentences).
    3.  The output MUST NOT contain emojis, hashtags, URLs, or any other non-text social media artifacts.

    Your response MUST be a JSON object conforming to the following schema:
    {DataOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(individual: Individual, reference_text: str) -> str:
    """
    Returns the user prompt containing all context from the individual.
    """
    return f"""
    Context:
    - Reference Text: "{reference_text}"
    - Role: "{individual['role']}"
    - Topic: "{individual['topic']}"

    Instruction:
    - Prompt: "{individual['prompt']}"

    Generate the data based on the instruction.
        """.strip()

# --- Main Agent Function ---
async def generate_data_for_individual(
    individual: Individual,
    reference_text: str,
    llm_agent: LLMAgent,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Generates the synthetic data for a single individual using its prompt.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(individual, reference_text)

    # Call the generic LLM agent and expect a DataOutput object
    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=DataOutput,
        temperature=temperature
    )

    if isinstance(response_obj, DataOutput):
        return response_obj.generated_text.strip()
    
    # Return None if the LLM call or validation failed
    return None