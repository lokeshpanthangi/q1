import os
import json
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer, pipeline

# Load environment variables from a .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# --- Part 1: Tokenization ---

def analyse_tokenization(name, tokenizer_path, sentence):
    """Tokenizes a sentence using a specified tokenizer and prints the analysis."""
    print(f"--- Tokenization Analysis for: {name} ---")
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Get tokens and their corresponding IDs
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Report the results
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Total Token Count: {len(tokens)}")
    print("-" * 50 + "\n")
    
    return {
        "tokens": tokens,
        "token_ids": token_ids,
        "count": len(tokens)
    }

# --- Part 2: Mask & Predict ---

def predict_masked_tokens(sentence):
    """
    Uses the model to predict masked tokens in the sentence.
    Returns predictions and saves them to predictions.json.
    """
    try:
        if '<mask>' not in sentence:
            print("Error: The sentence must contain at least one <mask> token.")
            return None

        # Initialize the pipeline
        fill_mask = pipeline(
            "fill-mask",
            model="distilroberta-base",  # You can replace this with your preferred model
            top_k=3
        )
        
        # Get predictions
        predictions = fill_mask(sentence)
        
        # Format predictions for JSON storage
        formatted_predictions = []
        for pred_set in predictions:
            if isinstance(pred_set, list):
                # Multiple masks case
                mask_predictions = []
                for p in pred_set:
                    mask_predictions.append({
                        "token": p["token"],
                        "token_str": p["token_str"].strip(),
                        "score": float(p["score"]),
                        "sequence": p["sequence"]
                    })
                formatted_predictions.append(mask_predictions)
            else:
                # Single mask case
                formatted_predictions.append({
                    "token": pred_set["token"],
                    "token_str": pred_set["token_str"].strip(),
                    "score": float(pred_set["score"]),
                    "sequence": pred_set["sequence"]
                })
        
        # Save predictions to JSON
        with open("predictions.json", "w") as f:
            json.dump(formatted_predictions, f, indent=4)
        
        return formatted_predictions
        
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

def display_predictions(predictions):
    """Display the predictions in a formatted way."""
    if predictions:
        print("\nPredictions saved to predictions.json")
        print("\nTop 3 predictions for each mask:")
        for i, pred_set in enumerate(predictions, 1):
            print(f"\nMask {i}:")
            if isinstance(pred_set, list):
                for j, pred in enumerate(pred_set[:3], 1):
                    print(f"{j}. {pred['token_str']} (probability: {pred['score']:.2f})")
            else:
                print(f"1. {pred_set['token_str']} (probability: {pred_set['score']:.2f})")

def main():
    """Main function to run the tokenization and prediction tasks."""
    
    while True:
        print("\n=== Project Menu ===")
        print("1: Run Analysis with Example Sentence")
        print("2: Run Analysis with Custom Sentence")
        print("3: Exit")
        choice = input("\nPlease enter your choice (1, 2, or 3): ")

        if choice == '1':
            # Part 1: Tokenization Analysis with example
            sentence_to_tokenize = "The cat sat on the mat because it was tired."
            print("\n--- Starting Tokenization Analysis ---")
            tokenizers_to_test = {
                "BPE (Byte-Pair Encoding)": "gpt2",
                "WordPiece": "bert-base-uncased",
                "Unigram": "t5-small"
            }
            
            tokenization_results = {}
            for name, path in tokenizers_to_test.items():
                tokenization_results[name] = analyse_tokenization(name, path, sentence_to_tokenize)
            
            # Part 2: Mask & Predict with example
            print("\n--- Starting Mask Prediction ---")
            masked_sentence = "The cat <mask> on the mat because it was <mask>."
            predictions = predict_masked_tokens(masked_sentence)
            display_predictions(predictions)

        elif choice == '2':
            # Get custom sentence for tokenization
            print("\nEnter a sentence for tokenization analysis:")
            custom_sentence = input("> ")
            
            print("\n--- Starting Tokenization Analysis ---")
            tokenizers_to_test = {
                "BPE (Byte-Pair Encoding)": "gpt2",
                "WordPiece": "bert-base-uncased",
                "Unigram": "t5-small"
            }
            
            tokenization_results = {}
            for name, path in tokenizers_to_test.items():
                tokenization_results[name] = analyse_tokenization(name, path, custom_sentence)
            
            # Get custom masked sentence
            print("\nNow enter the same sentence with <mask> tokens where you want predictions:")
            print("Example: 'The <mask> ran to the <mask>.'")
            masked_sentence = input("> ")
            
            print("\n--- Starting Mask Prediction ---")
            predictions = predict_masked_tokens(masked_sentence)
            display_predictions(predictions)

        elif choice == '3':
            print("\nExiting program. Goodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 