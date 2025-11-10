# metrics/diversity.py
import zlib

def calculate_compression_ratio(text: str) -> float:
    """
    Calculates the Compression Ratio of a text.
    Based on the premise that text with high redundancy
    (low lexical diversity) is easier to compress.

    Ratio = Original Size / Compressed Size

    A HIGHER value indicates HIGHER redundancy (worse diversity).
    A value closer to 1.0 indicates LOW redundancy (better diversity).
    """
    if not text:
        return 1.0  # An empty text has no redundancy

    try:
        original_size = len(text.encode('utf-8'))
        if original_size == 0:
            return 1.0
            
        compressed_size = len(zlib.compress(text.encode('utf-8')))
        if compressed_size == 0:
            return 1.0 # Avoid division by zero
            
        return original_size / compressed_size
        
    except Exception:
        return 1.0 # Safe fallback

def calculate_internal_repetition(
    text: str,
    n_min: int = 4,
    n_max: int = 6
) -> float:
    """
    Measures Internal Self-Repetition, an adaptation of Self-Repetition
    for a single document.

    It looks for repeated n-grams ("template phrases") in a range
    adapted for microblogging (n=4 to 6).

    It calculates this as the "repetition rate":
    Rate = (Total N-grams - Unique N-grams) / Total N-grams
    
    A value of 0.0 is perfect (0% repetition, high diversity).
    A value of 1.0 would be the theoretical worst (all repetitions).
    """
    all_ngrams = []
    words = text.lower().split()
    
    # Generate all n-grams in the [n_min, n_max] range
    for n in range(n_min, n_max + 1):
        if len(words) < n:
            continue # Skip if text is shorter than n
        for i in range(len(words) - n + 1):
            all_ngrams.append(" ".join(words[i:i+n]))
            
    if not all_ngrams:
        return 0.0  # 0% repetition if no n-grams were found
        
    total_ngrams = len(all_ngrams)
    unique_ngrams = len(set(all_ngrams))
    
    # This is the rate of n-grams that are repetitions.
    repetition_rate = (total_ngrams - unique_ngrams) / float(total_ngrams)
    return repetition_rate