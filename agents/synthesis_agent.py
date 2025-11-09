# agents/synthesis_agent.py
from pydantic import BaseModel, Field
from typing import Optional, Tuple
from .llm_agent import LLMAgent

# --- Pydantic Output Model ---
class SynthesisOutput(BaseModel):
    """
    Defines the expected JSON structure for the SynthesisAgent's output.
    It must contain both the topic and the prompt.
    """
    topic: str = Field(
        ...,
        description="A concise topic that captures the core theme, based on the analysis."
    )
    prompt: str = Field(
        ...,
        description="A high-quality instruction (prompt) to guide another LLM, aligned with the role, topic, and text."
    )

# --- System Prompt ---
def _get_system_prompt() -> str:
    """
    Returns the Chain-of-Thought (CoT) system prompt for the Synthesis Agent .
    """
    return f"""
    You are an expert prompt engineer. Your task is to generate a Topic and a Prompt based on a given context.
    You must follow this internal reasoning process:

    1.  **Analyze**: Read the provided Reference Text and Role to understand the situation, tone, and perspective.
    2.  **Define Topic**: Based on your analysis, define a concise Topic that captures the core theme.
    3.  **Construct Prompt**: Using the Topic you just defined and your analysis, construct a high-quality instruction (prompt) that guides another LLM to generate a text. This prompt must be aligned with the Role, the Topic, and the Reference Text.

    Your response MUST be a single JSON object conforming to the following schema:
    {SynthesisOutput.model_json_schema()}
        """.strip()

# --- User Prompt ---
def _get_user_prompt(role: str, reference_text: str) -> str:
    """
    Returns the user prompt containing the role and reference text [cite: 830-832].
    """
    return f"""
    Reference Text: "{reference_text}"
    Role: "{role}"

    Generate the Topic and Prompt based on this context.
        """.strip()

# --- Main Agent Function ---
async def generate_topic_and_prompt(
    role: str,
    reference_text: str,
    llm_agent: LLMAgent,
    temperature: float = 0.7
) -> Optional[Tuple[str, str]]:
    """
    Dynamically generates a topic and a prompt from a role and reference text.
    """
    system_prompt = _get_system_prompt()
    user_prompt = _get_user_prompt(role, reference_text)

    # Call the generic LLM agent and expect a SynthesisOutput object
    response_obj = await llm_agent.call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_model=SynthesisOutput,
        temperature=temperature
    )

    if isinstance(response_obj, SynthesisOutput):
        # Return both fields as a tuple
        return response_obj.topic, response_obj.prompt
    
    # Return None if the LLM call or validation failed
    return None