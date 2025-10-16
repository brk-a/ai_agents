# BYO LLM

## 1. build a bigram model
### 1.1. WTF s a bigram?
* a sequence of two adjacent elements from a string of tokens
* tokens are usually letters, syllables or words
* in the context of language and large language models (LLMs), bigrams refer to pairs of consecutive words or characters used to analyse and predict text patterns
* bigram analysis is a basic form of n-gram modelling where n=2
    - helps in understanding language structure, building language models for tasks like speech recognition, text prediction and cryptography
* example: the phrase **"big data"** is a bigram because it is a pair of two adjacent words
### 1.2. how TF  are bigrams used in Language Models (LMs) like GPT?
* bigrams are  used in simple statistical language models to capture local word dependencies
* modern LLMs like GPT rely on more complex transformer architectures considering long-range context; bigrams form a foundational concept showing how word sequences can be probabilistically modelled
* in these simple models, the probability of a word depends only on the immediately preceding word which helps predict or generate text based on observed patterns in training data
### 1.3. how TF to compute bigram probabilities from a corpus
* to compute bigram probabilities, walk through the corpus to count occurrences of each word (unigrams) and each pair of consecutive words (bigrams)
* the bigram probability P(word₂ | word₁) is then estimated as the count of the pair (word₁, word₂) divided by the count of word₁ alone
* formally:
    
    $$ P(w_2|w_1) = \frac{C(w_1, w_2)}{C(w_1)} $$

    where $$C(w_1, w_2)$$ is bigram count
    
    and $$C(w_1)$$ is unigram count
    
* this simple approach enables prediction of the next word given the current one

### 1.4. WTF are the practical limits of bigram models for text generation?
* bigram models are easy to implement but limited because they consider only immediate word pairs, ignoring longer context and linguistic structures
* they often assign zero probability to unseen word pairs making them brittle without smoothing techniques
* increasing `n` (to trigrams or higher) improves context capture but leads to computational challenges and data sparsity
* modern LLMs overcome these limits by using deep learning and attention mechanisms to capture long-range dependencies and richer semantic information for natural text generation

### 1.5 tensors instead of arrays
* arrays are not scalable; we need more complex but efficient data structures
* presenting the tensor...
    - every come accross the following: scalar, vector and matrix?
    - a scalar can be represented as an `int` or `float`: a single number you can multiply (aka scale) anything by
    - a vector can be represented as an array: forget the direction bit of a vector for now
    - a matrix can be represented as a 2D array: a list of lists; a vector made up of vectors
    - a tensor if a matrix of matrices: a 3D array if you like
