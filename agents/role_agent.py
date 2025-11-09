# agents/role_agent.py
from pydantic import BaseModel, Field
from typing import Optional
from .llm_agent import LLMAgent

# --- Pydantic Output Model ---
class RoleOutput(BaseModel):
    """
    Defines the expected JSON structure for the RoleAgent's output.
    """
    role: str = Field(
        ...,
        description="The inferred speaker role based on the text (e.g., a concerned citizen, a health expert, an official spokesperson)."
    )

# --- System Prompt ---
def _get_system_prompt() -> str:
    """
    Returns the system prompt for the Role Agent
    """
    return f"""
    You are an expert text analyst. Your task is to accurately infer the
    most likely role of the speaker given a short piece of text.
    Focus on the context, tone, and content to determine their perspective.

    Your response MUST be a JSON object conforming to the following schema:
    {RoleOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(reference_text: str) -> str:
    """
    Returns the user prompt containing the reference text
    """
    return f"Reference Text: \"{reference_text}\"\n\nInfer the speaker's role based on this text."

# --- Main Agent Function ---
async def infer_role(
    reference_text: str, 
    llm_agent: LLMAgent,
    temperature: float = 0.5
) -> Optional[str]:
    """
    Dynamically infers a speaker role from the reference text.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(reference_text)

    # Call the generic LLM agent and expect a RoleOutput object
    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=RoleOutput,
        temperature=temperature
    )

    if isinstance(response_obj, RoleOutput):
        return response_obj.role
    
    # Return None if the LLM call failed or validation failed
    return None