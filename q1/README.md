# Project: Understanding Tokenization & Masked Language Models

## Overview

This project is a hands-on exploration of two fundamental concepts in modern Natural Language Processing (NLP): **Tokenization** and **Masked Language Modeling**.

The interactive Python script allows you to:
1.  Analyze how three different tokenization algorithms (BPE, WordPiece, Unigram) break down the same sentence.
2.  Use a state-of-the-art language model interactively to perform a "fill-in-the-blank" task on any sentence you provide.

---

## 🧠 Key Learnings & Concepts

This project was designed to provide practical insights into what happens "under the hood" of large language models.

### 1. Tokenization: The Art of Splitting Text

Tokenization is the process of converting a sequence of text into smaller pieces called tokens. This is a critical first step for any language model. We explored three dominant algorithms:

-   **BPE (Byte-Pair Encoding)**: Used by models like GPT-2. It starts with individual characters and greedily merges the most frequent adjacent pairs of tokens to build its vocabulary. This is why we see the `Ġ` character (representing a space) in the `gpt2` output, as it helps reconstruct the original text perfectly.

-   **WordPiece**: Used by models like BERT. It's similar to BPE but makes its merging decision based on maximizing the likelihood of the training data, not just frequency. It often breaks words into a primary "root" word and subsequent pieces marked with `##` (though not seen in our simple example).

-   **Unigram (SentencePiece)**: Used by models like T5 and ALBERT. It takes a different, probabilistic approach. It starts with a large vocabulary and progressively removes tokens, keeping those that are most essential for reconstructing the text. It's powerful because it handles whitespace natively, which is why we see a ` ` (space) character at the beginning of most tokens.

**Key Takeaway**: The same sentence results in different tokens and IDs across algorithms. This choice impacts the model's vocabulary size, its ability to handle unknown words, and ultimately its performance.

### 2. Masked Language Modeling (MLM)

We used `distilbert/distilroberta-base` to predict masked words. This is a **Masked Language Model**, which is different from a Causal Language Model (like GPT).

-   **Causal LM (e.g., GPT)**: Is trained to predict the *next* word in a sequence. It's great for text generation.
-   **Masked LM (e.g., BERT, RoBERTa)**: Is trained to predict words that have been deliberately hidden (masked) within a sentence. This gives it a deep, bidirectional understanding of context, making it perfect for tasks like "fill-in-the-blank."

**Key Takeaway**: Choosing the right model architecture is crucial. For understanding context and filling in missing information, an MLM is the superior choice.

### 3. Using the Hugging Face Inference API

Instead of downloading and running a large model locally (which requires significant computational resources), we used the Hugging Face Inference API.

**Key Takeaway**: APIs provide a powerful and efficient way to leverage state-of-the-art models without the hardware overhead. Securely using them with API tokens (`.env` file) is a standard best practice.

---

## 🚀 How to Use the Project

### Setup

1.  **Install Dependencies**: Make sure you are in the project's root directory and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure API Token**:
    -   Create a file named `.env` in the project root.
    -   Get a free "read" access token from [Hugging Face settings](https://huggingface.co/settings/tokens).
    -   Add it to the `.env` file like this: `HF_API_TOKEN="hf_YourTokenGoesHere"`

### Running the Script

1.  Execute the script from the root directory:
    ```bash
    python q1/tokenise.py
    ```

2.  **Use the Interactive Menu**:
    -   **Choice `1`**: Runs the static tokenization analysis on the predefined sentence.
    -   **Choice `2`**: Enters the interactive "Fill-in-the-Blank" mode. You will be prompted to enter your own sentence with one or more `<mask>` tokens.
    -   **Choice `3`**: Exits the program. 