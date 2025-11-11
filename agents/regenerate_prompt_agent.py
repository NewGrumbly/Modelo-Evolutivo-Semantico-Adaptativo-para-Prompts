# agents/regenerate_prompt_agent.py
from pydantic import BaseModel, Field
from typing import Optional
from .llm_agent import LLMAgent

# --- Pydantic Output Model ---
class RegeneratePromptOutput(BaseModel):
    """
    Defines the JSON structure for the RegeneratePromptAgent's output.
    """
    prompt: str = Field(
        ...,
        description="The new, high-quality instruction (prompt) generated from the evolved role and topic."
    )

# --- System Prompt ---
def _get_system_prompt() -> str:
    """
    Returns the system prompt for the Regenerate Prompt Agent.
    This prompt instructs the LLM to build a new prompt from
    the evolved genome.
    """
    return f"""
    You are an expert prompt engineer. Your task is to write a single,
    high-quality instruction (prompt) based on a given Role, Topic,
    and Reference Text.

    The prompt must be perfectly aligned with all three inputs and
    guide another LLM to generate a short, 1-2 sentence text.
        """.strip()

# --- User Prompt ---
def _get_user_prompt(role: str, topic: str, reference_text: str) -> str:
    """
    Returns the user prompt containing the evolved genome
    and the reference text.
    """
    return f"""
    - Reference Text: "{reference_text}"
    - Role: "{role}"
    - Topic: "{topic}"

    Generate the instruction (prompt) based on these components.
        """.strip()

# --- Main Agent Function ---
async def regenerate_prompt(
    role: str,
    topic: str,
    reference_text: str,
    llm_agent: LLMAgent,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Generates a new prompt from an evolved role and topic.
    Returns the new prompt string or None on failure.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(role, topic, reference_text)

    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=RegeneratePromptOutput,
        temperature=temperature
    )

    if isinstance(response_obj, RegeneratePromptOutput):
        return response_obj.prompt.strip()
    
    return None