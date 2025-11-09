# prepare_corpus.py
import csv
from pathlib import Path
import sys

# --- Configuration ---
# Reads from the data/ directory
INPUT_FILE = Path("data/corpus.csv")
# Writes to the root directory
OUTPUT_FILE = Path("data/filtered_corpus.csv")
MIN_WORDS = 5 # Minimum word threshold
# ---------------------

def filter_simple_corpus():
    """
    Reads the input CSV, filters by word count,
    and saves to the output CSV.
    """
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Please place your 'corpus.csv' file in the 'data/' directory.")
        sys.exit(1)

    print(f"Starting filtering of {INPUT_FILE} (Minimum: {MIN_WORDS} words)...")
    
    read_count = 0
    saved_count = 0

    try:
        # Open both files
        with open(INPUT_FILE, mode='r', encoding='utf-8') as f_in, \
             open(OUTPUT_FILE, mode='w', encoding='utf-8', newline='') as f_out:
            
            # Input CSV has no header
            reader = csv.reader(f_in)
            
            # Output CSV will also have no header
            writer = csv.writer(f_out)

            for row in reader:
                read_count += 1
                if not row:  # Skip empty rows
                    continue
                
                # Take the text from the single column
                tweet_text = row[0].strip()
                
                # 1. Count words (split by spaces)
                words = tweet_text.split()
                
                # 2. Apply filter
                if len(words) >= MIN_WORDS:
                    # Save the original row
                    writer.writerow(row) 
                    saved_count += 1
                
                if read_count % 500000 == 0:
                    print(f"  ... Processed {read_count:,} tweets...")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

    print("\n--- Filtering Complete ---")
    print(f"Tweets read:    {read_count:,}")
    print(f"Tweets saved: {saved_count:,} (>= {MIN_WORDS} words)")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_simple_corpus()