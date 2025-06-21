import os
import json
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer

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

# --- Part 2: Mask & Predict ---

def query_fill_mask_api(masked_sentence, model_id="distilbert/distilroberta-base"):
    """
    Queries the Hugging Face Inference API to predict masked tokens in a sentence.
    Prints the completed sentence and saves the raw JSON output.
    """
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    print(f"--- Querying Fill-Mask API for: {model_id} ---")
    print(f"Input Sentence: {masked_sentence}")

    payload = {"inputs": masked_sentence}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        predictions = response.json()

        output_path = os.path.join("q1", "predictions.json")
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
            
        print(f"Successfully received predictions and saved to {output_path}")

        # --- Fill in the blanks and display the result ---
        completed_sentence = masked_sentence
        if isinstance(predictions, list) and predictions:
            # Handle multiple masks, where the API returns a list of lists
            if isinstance(predictions[0], list):
                for blank_predictions in predictions:
                    if blank_predictions:
                        top_prediction = blank_predictions[0]['token_str']
                        completed_sentence = completed_sentence.replace("<mask>", top_prediction, 1)
            # Handle a single mask, where the API returns a list of dicts
            else:
                top_prediction = predictions[0]['token_str']
                completed_sentence = completed_sentence.replace("<mask>", top_prediction, 1)

            print("\n--- Prediction Result ---")
            print(f"Completed Sentence: {completed_sentence}\n")
        
        return predictions

    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        if "401" in str(e):
            print("Authentication error (401): Please ensure your HF_API_TOKEN is correct.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def interactive_fill_mask():
    """Handles the user interaction for the fill-mask feature."""
    while True:
        print("\nEnter a sentence with one or more <mask> tokens to get predictions.")
        print("(Type 'back' to return to the main menu)")
        user_input = input("> ")

        if user_input.lower() == 'back':
            break
        
        if '<mask>' not in user_input:
            print("Error: Your sentence must include at least one '<mask>' token.")
            continue

        query_fill_mask_api(user_input)

def main():
    """Main function to run the tokenization and prediction tasks."""
    
    while True:
        print("--- Project Menu ---")
        print("1: Run Tokenization Analysis")
        print("2: Interactive Fill-in-the-Blank")
        print("3: Exit")
        choice = input("Please enter your choice (1, 2, or 3): ")

        if choice == '1':
            sentence_to_tokenize = "The cat sat on the mat because it was tired."
            tokenizers_to_test = {
                "BPE (Byte-Pair Encoding)": "gpt2",
                "WordPiece": "bert-base-uncased",
                "Unigram": "t5-small"
            }
            
            print("\n--- Starting Tokenization Analysis ---")
            for name, path in tokenizers_to_test.items():
                analyse_tokenization(name, path, sentence_to_tokenize)
        
        elif choice == '2':
            if not HF_API_TOKEN:
                print("\nError: HF_API_TOKEN not found.")
                print("Please create a .env file with your Hugging Face API token to use this feature.")
            else:
                interactive_fill_mask()
        
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 