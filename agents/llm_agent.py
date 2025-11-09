# agents/llm_agent.py
import ollama
from typing import Optional, Type
from pydantic import BaseModel

class LLMAgent:
    """
    Centralized asynchronous class to handle all interactions with the LLM model.
    """
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.client = ollama.AsyncClient(host='http://127.0.0.1:11434')
    
    async def call_llm(
        self, 
        system_prompt: str,
        user_prompt: str,
        output_model: Type[BaseModel], # The specific agent tells us what Pydantic model to expect
        temperature: float = 0.7
    ) -> Optional[BaseModel]:
        """
        Generic method to call the LLM, force JSON output, and validate the output.
        """
        try:
            response = await self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": temperature}
            )
            
            content = response["message"]["content"]

            # Validate the JSON string returned bt the LLM
            if isinstance(content, str):
                return output_model.model_validate_json(content)
            
            # Sometimes it might return a dict directly
            if isinstance(content, dict):
                return output_model.model_validate(content)
            
            raise ValueError("Unexpected response type from LLM (not str or dict)")

        except Exception as e:
            # Generic error logging
            print(f"❌ Error in LLMAgent.call_llm (Model: {output_model.__name__}): {e}")
            print(f"   Input System Prompt: {system_prompt[:100]}...")
            print(f"   Input User Prompt: {user_prompt[:100]}...")
            return None