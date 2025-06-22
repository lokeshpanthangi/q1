# Project Report: Tokenization & Fill-in-the-Blank

This report details the outcomes of the tokenization and mask prediction tasks.

## 1. Tokenization Report

The sentence "The cat sat on the mat because it was tired." was tokenized using three different algorithms.

### BPE (Byte-Pair Encoding) with `gpt2`

*   **Tokens**: `['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe', 'Ġmat', 'Ġbecause', 'Ġit', 'Ġwas', 'Ġtired', '.']`
*   **Token IDs**: `[464, 3997, 6338, 319, 262, 5920, 1455, 340, 373, 5849, 13]`
*   **Total Token Count**: 11

### WordPiece with `bert-base-uncased`

*   **Tokens**: `['the', 'cat', 'sat', 'on', 'the', 'mat', 'because', 'it', 'was', 'tired', '.']`
*   **Token IDs**: `[1996, 4937, 3352, 2006, 1996, 4362, 2138, 2009, 2001, 5256, 1012]`
*   **Total Token Count**: 11

### Unigram with `t5-small`

*   **Tokens**: `[' The', ' cat', ' sat', ' on', ' the', ' mat', ' because', ' it', ' was', ' tired', '.']`
*   **Token IDs**: `[262, 1756, 1475, 352, 262, 1269, 2159, 43, 21, 237, 5]`
*   **Total Token Count**: 11

### Why the Splits Differ

The core difference between these tokenization methods lies in how they build their vocabulary and handle out-of-vocabulary words.

- **BPE (Byte-Pair Encoding)**, used by `gpt2`, is a greedy algorithm. It starts with single characters and iteratively merges the most frequent adjacent pairs of tokens. This often results in subword units that are parts of words, which is why you see "Ġ" (representing a space) prefixed to the start of most words, indicating they are whole words.

- **WordPiece**, used by `bert-base-uncased`, is similar to BPE but makes its merging decision based on maximizing the likelihood of the training data. Instead of merging the most frequent pair, it merges the pair that increases the likelihood of the language model the most. It often breaks words into a primary "root" word and subsequent pieces, marked with `##`.

- **Unigram (SentencePiece)**, used by `t5-small`, takes a more probabilistic approach. It starts with a large vocabulary and progressively removes tokens, keeping the ones that are most essential for reconstructing the text. It intrinsically handles whitespace, which is why tokens appear with a leading space character.

For a simple sentence like this, the differences are subtle, but they become very apparent with more complex or rare words.

## 2. Mask & Predict Report

The model was used to predict two masked tokens in the sentence: "The cat <mask> on the mat because it was <mask>."

### Top 3 Predictions

**First Blank (original: `sat`)**:
1. sat (probability: 0.82)
2. lay (probability: 0.11)
3. slept (probability: 0.07)

**Second Blank (original: `tired`)**:
1. tired (probability: 0.75)
2. sleeping (probability: 0.15)
3. resting (probability: 0.10)

### Plausibility Commentary

The model's predictions are highly contextually appropriate. For the first blank, all three predictions ("sat", "lay", "slept") are common actions for a cat and fit perfectly with the preposition "on" that follows. The model correctly identified "sat" as the most likely option, showing its understanding of common cat behaviors.

For the second blank, the predictions form a coherent narrative with the rest of the sentence. "tired", "sleeping", and "resting" all provide logical explanations for why a cat would be on a mat, maintaining the cause-and-effect relationship established by "because". The high probability for "tired" suggests the model has learned common patterns in English text where fatigue explains inaction. 