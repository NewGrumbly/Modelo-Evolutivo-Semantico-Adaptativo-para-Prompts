# utils/setup.py
import random
import csv
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

# Rutas de los corpus (actualizadas)
CORPUS_FILE = Path("../data/filtered_corpus.csv")
EXAMPLE_CORPUS_FILE = Path("../data/example_corpus.csv")

def load_random_reference(corpus_arg: Optional[str] = None) -> str:
    """
    Loads a random line from the specified corpus or the default one.
    """
    file_to_load = CORPUS_FILE
    
    if corpus_arg:
        file_to_load = Path(corpus_arg)
    
    if not file_to_load.exists():
        if corpus_arg:
            print(f"Warning: Specified corpus '{corpus_arg}' not found.")
        
        if EXAMPLE_CORPUS_FILE.exists():
            print(f"Warning: Using '{EXAMPLE_CORPUS_FILE}' as fallback.")
            file_to_load = EXAMPLE_CORPUS_FILE
        else:
            raise FileNotFoundError(
                f"No corpus file found. Looked for '{file_to_load}' and '{EXAMPLE_CORPUS_FILE}'.\n"
                "Please run 'python prepare_corpus.py' or add 'data/example_corpus.csv'."
            )
            
    # Read CSV file
    with open(file_to_load, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [row[0] for row in reader if row and row[0].strip()]
    
    if not lines:
        raise ValueError(f"Corpus file '{file_to_load}' is empty.")
    
    return random.choice(lines)

def setup_experiment(
    base_dir: Path,
    reference_text_arg: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Prepares the environment for a single GA run.
    1. Creates a unique timestamped directory inside 'base_dir'.
    2. Loads the reference text.
    3. Saves a copy of the reference text inside the new directory.
    (Adapted from)
    """
    
    # 1. Create unique output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load reference text
    reference_text = ""
    if reference_text_arg:
        p_ref = Path(reference_text_arg)
        if not p_ref.exists():
            raise FileNotFoundError(f"Specified reference file not found: {reference_text_arg}")
        reference_text = p_ref.read_text(encoding="utf-8").strip()
    else:
        # Load a random one from the corpus
        reference_text = load_random_reference()
    
    # 3. Save the reference text to the output directory
    ref_save_path = output_dir / "reference_text.txt"
    ref_save_path.write_text(reference_text, encoding="utf-8")
    
    return output_dir, reference_text