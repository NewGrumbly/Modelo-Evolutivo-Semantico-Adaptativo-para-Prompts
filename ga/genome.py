# ga/genome.py
from typing import TypedDict, Optional

"""
Defines the genome for an individual in the population.
"""

class Individual(TypedDict):
    # Attributes evolved by semantic operators
    role: str
    topic: str
    
    # Attributes generated from the evolved genome
    prompt: str
    generated_data: Optional[str]

    # Evaluation metric
    fitness: float